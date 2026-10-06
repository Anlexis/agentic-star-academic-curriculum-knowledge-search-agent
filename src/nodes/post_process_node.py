"""AgentCore Platform v1.0"""

# EDU-C2-005 - PostProcessNode (outer post_process slot; output security gate)
#
# Reads the final knowledge-base answer from state["result"], which is
# populated by CurriculumKnowledgeGraphNode.merge_output() (mapped from the
# inner graph's formatted_answer output), and surfaces it as the finalized
# output AFTER running the output content-safety gate and enforcing the
# documented output schema.
#
# Gate design: this node calls the MODULE-LEVEL `_security_gate_output()`
# scan from execute() itself. The gate accepts `Any` (not just `str`) and
# RECURSES into dict/list/tuple/set so a violation nested inside a structured
# payload (e.g. a citation dict) cannot slip past a top-level-only string
# scan. Output is scanned for disallowed content (API keys, JWT tokens,
# Bearer tokens, raw credential assignments) - generic, domain-agnostic
# patterns. On a violation, formatted_output is replaced with a sanitised
# stub and ERROR status is returned. No _extra_security_gate_input/_output
# instance methods are defined on this node (the framework auto-wraps such
# hooks).
#
# Schema enforcement: the documented output contract states that EVERY answer
# this template emits carries the standing advisory disclaimer. Composition
# guarantees it today (OutputFormatNode appends it), but this gate
# independently ENFORCES it - a future rendering change cannot silently drop
# the disclaimer, because this node re-appends it (with an audit event) to
# any clean string output that lacks it.
#
# This module-level `_security_gate_output()` is also imported and reused by
# `AcademicCurriculumKnowledgeAgent.get_output()` (src/graph/graph.py) before
# it surfaces the structured `citations` field - the same gate, applied a
# second time at the outer envelope boundary, fail-closed.

import logging
import re
from typing import Any, ClassVar, List, Optional, Tuple

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_state import AgentState
from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel
from shared.utils.audit_logger import emit_trace_event
from src.services.progress import emit_progress
from src.services.failure_message import EMPTY_INPUT, INPUT_REJECTED, INVALID_VALUE, OUTPUT_BLOCKED
from src.nodes.output_format_node import _ADVISORY_DISCLAIMER

logger = logging.getLogger(__name__)

# Disallowed content patterns. Each tuple: (name, compiled regex) -
# order matters (most specific first).
_DISALLOWED_PATTERNS: List[Tuple[str, "re.Pattern[str]"]] = [
    # API key patterns: sk-..., pk-..., ak-...
    ("api_key", re.compile(r"\b(?:sk|pk|ak)-[A-Za-z0-9]{16,}", re.IGNORECASE)),
    # JWT: three base64url segments separated by dots
    ("jwt", re.compile(r"eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}")),
    # Bearer token in Authorization-like context
    ("bearer_token", re.compile(r"Bearer\s+[A-Za-z0-9._~+/]{20,}", re.IGNORECASE)),
    # Credential assignment patterns
    (
        "credential_assignment",
        re.compile(
            r"\b(?:password|passwd|secret|api_key|token|access_key|private_key)\s*[:=]\s*\S{8,}",
            re.IGNORECASE,
        ),
    ),
]

_SANITISED_STUB = (
    "[OUTPUT BLOCKED by the output security gate - disallowed content "
    "detected. Review the generated output and retry without "
    "credential-like strings.]"
)

# Recursion depth cap - defensive against pathological/self-referential
# structures; a legitimate agent payload never nests this deep.
_MAX_SCAN_DEPTH = 12


def _scan_value(value: Any, depth: int = 0) -> Optional[str]:
    """Recursively scan a value for disallowed content.

    Handles str directly; recurses into dict (values only - keys are
    schema/field names, not caller-influenced content), list, tuple, and
    set. Any other scalar (int/float/bool/None/etc.) carries no credential
    risk and is skipped. Returns the name of the first violation found, or
    None if clean.
    """
    if depth > _MAX_SCAN_DEPTH:
        return None
    if isinstance(value, str):
        for name, pattern in _DISALLOWED_PATTERNS:
            if pattern.search(value):
                return name
        return None
    if isinstance(value, dict):
        for v in value.values():
            hit = _scan_value(v, depth + 1)
            if hit:
                return hit
        return None
    if isinstance(value, (list, tuple, set)):
        for v in value:
            hit = _scan_value(v, depth + 1)
            if hit:
                return hit
        return None
    return None


def _security_gate_output(content: Any) -> Optional[str]:
    """Run the output content gate.

    Accepts Any (str, dict, list, or a mix thereof) so a violation buried
    inside a structured payload is still caught - never a top-level-only
    string scan. Returns the name of the first matched violation, or None
    if clean.
    """
    return _scan_value(content)


def _enforce_disclaimer(result: str, state: AgentState) -> str:
    """Independently enforce the documented always-on advisory disclaimer.

    OutputFormatNode composes the disclaimer into every answer; this gate
    re-checks the invariant on whatever string actually reaches it and
    repairs a missing disclaimer (with an audit event) rather than trusting
    upstream composition.
    """
    if _ADVISORY_DISCLAIMER in result:
        return result
    emit_trace_event(
        "output_disclaimer_enforced",
        {"output_chars": len(result)},
        state,
    )
    return f"{result}\n\n---\n\n*{_ADVISORY_DISCLAIMER}*"


# Caller-facing wording for a run that completed without an answer. The marker
# is an internal reason code; this maps it to the sentence the caller sees.
# Static sentences only - no request value is ever substituted, so nothing the
# caller sent can be reflected back through this path.
_DEGRADED_MESSAGES = {
    "EMPTY_INPUT": EMPTY_INPUT,
    "INVALID_REQUEST": INVALID_VALUE,
}


class PostProcessNode(FunctionNode):
    """Output gate: scan the final KB answer and enforce the output schema.

    Outer backbone post_process slot. Reads state["result"] (the merged
    formatted_answer from CurriculumKnowledgeGraphNode.merge_output()),
    applies the content-safety gate, and enforces the standing advisory
    disclaimer before the response is returned to the caller.

    Input state keys:
        result: final formatted KB answer (from merge_output)

    Output state keys (partial dict):
        formatted_output: sanitised output (gated answer if clean;
                          blocked stub on violation)
        result:           gated alongside formatted_output (blocked stub on
                          violation; disclaimer-enforced on the clean path;
                          forwarded untouched when empty)
        status:           AgentStatus.SUCCESS.value or AgentStatus.ERROR.value
                          (plain strings - never the bare enum in State)
        error_log:        (on error) list of error messages
    """

    # Explicit by design, not inherited implicitly. Outer backbone gate slot -
    # matches the manifest's declared required_trust_level (config/agent.yaml).
    required_trust_level: ClassVar[TrustLevel] = TrustLevel.VERIFIED_EXTERNAL

    def execute(self, state: AgentState) -> dict[str, Any]:
        emit_progress("Finalising the response...")

        # The run completed without an answer because the request could not be
        # accepted as written. Report the reason as the response: the caller
        # needs to know what to change, and an empty body would leave them with
        # nothing. Status stays SUCCESS - the run did what it could with the
        # request it was given, and the caller can correct it and send again on
        # the same conversation.
        marker = state.get("error_code")
        if marker:
            message = _DEGRADED_MESSAGES.get(marker, INPUT_REJECTED)
            emit_trace_event(
                "post_process_degraded",
                {"reason": marker},
                state,
            )
            return {
                "formatted_output": message,
                "result": message,
                "status": AgentStatus.SUCCESS.value,
                "error_code": marker,
            }

        result = state.get("result", "")

        if not result or (isinstance(result, str) and not result.strip()):
            # No result to gate - forward as-is (non-fatal).
            return {
                "formatted_output": result,
                "status": AgentStatus.SUCCESS.value,
            }

        violation = _security_gate_output(result)
        if violation:
            logger.error(
                "PostProcessNode: OUTPUT BLOCKED - violation type: %s",
                violation,
            )
            emit_progress(OUTPUT_BLOCKED)
            return {
                "formatted_output": _SANITISED_STUB,
                "result": _SANITISED_STUB,
                "status": AgentStatus.ERROR.value,
                "error_log": [f"PostProcessNode: output blocked - " f"disallowed content detected ({violation})"],
            }

        # Domain output formatting (body + sources + advisory disclaimer) is
        # composed upstream by the inner graph's OutputFormatNode; `result`
        # here is already the final rendered string
        # (CurriculumKnowledgeGraphNode.merge_output() maps the inner
        # formatted_answer -> outer `result`). This node gates and enforces -
        # it does not compose, and it does not reach into raw
        # request/response dicts (fail-closed whitelisting).
        if isinstance(result, str):
            result = _enforce_disclaimer(result, state)

        # Clean - domain audit: a finalized output was emitted.
        emit_trace_event(
            "post_process_complete",
            {"output_chars": len(str(result))},
            state,
        )

        return {
            "formatted_output": result,
            "result": result,
            "status": AgentStatus.SUCCESS.value,
        }
