"""AgentCore Platform v1.0"""

# EDU-C2-005 - PreProcessNode (outer pre_process slot; input validation +
# identifier screen)
#
# Node contract: extend FunctionNode; implement execute(self, state) -> dict -
# no `config` parameter. Return ONLY the fields this node changes (never full
# state). Read input_context via state.get("input_context", {}) - read-only.
# Never import from mediator/, api/, or other agents.
#
# Trust gate: this is an outer backbone gate slot - the manifest declares
# required_trust_level: "VERIFIED_EXTERNAL" (config/agent.yaml), so this node
# gates external callers before the inner domain workflow runs.
# Identifier screen: a surface screen redacts long numeric ID sequences and
# e-mail addresses from the free-text query payload before validated_input is
# written, so raw identifiers never reach the inner domain nodes or the
# checkpoint DB. The KB this template searches is regulatory/administrative
# content only (MEXT / NIAD-QE / institutional policy text) - no student
# records are ever read or stored; this screen is defense-in-depth on the
# caller's free-text question, not a student-data path.

import re
from typing import Any, ClassVar, List

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_state import AgentState
from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel
from shared.utils.audit_logger import emit_trace_event
from src.services.progress import emit_progress
from src.services.failure_message import EMPTY_INPUT

# Surface-level identifier patterns redacted before validated_input is
# written. Downstream domain nodes only ever operate on the normalised
# question text and KB passage summaries, never a raw ID a caller might paste
# in while describing their situation (e.g. "my student ID is ...").
_PII_PATTERNS: List[re.Pattern[str]] = [
    # Long numeric ID sequences: 10-19 consecutive digits (optionally grouped).
    re.compile(r"\b\d{4}[- ]?\d{4}[- ]?\d{2,11}\b"),
    # E-mail addresses.
    re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
]
_PII_REPLACEMENT = "[REDACTED]"

# Caller-supplied telemetry labels (channel) are locked to an inert
# identifier shape before they are written to state - free text in a
# caller-controlled label is a log/output injection vector.
_INERT_IDENTIFIER_RE = re.compile(r"^[a-z0-9_]{1,32}$")


def _surface_strip_identifiers(text: str) -> str:
    """Redact obvious direct-identifier tokens from a free-text string."""
    for pattern in _PII_PATTERNS:
        text = pattern.sub(_PII_REPLACEMENT, text)
    return text


def _inert_label(value: object) -> str:
    """Lock a caller-supplied label to an inert identifier, else 'unknown'.

    The rejected value is never echoed anywhere - it is simply replaced.
    """
    if isinstance(value, str) and _INERT_IDENTIFIER_RE.match(value):
        return value
    return "unknown"


# Marker for a run that completes without an answer because no question was
# supplied. The caller corrects this by sending a question, so it is a
# completion with a reason rather than a terminal failure.
_CODE_EMPTY_INPUT = "EMPTY_INPUT"


class PreProcessNode(FunctionNode):
    """Input validation + identifier screen before main processing.

    Rejects empty / invalid input before the inner domain workflow graph
    runs, and surface-strips direct identifiers from the payload.
    """

    # Explicit by design, not inherited implicitly. Outer backbone gate slot -
    # matches the manifest's declared required_trust_level (config/agent.yaml).
    required_trust_level: ClassVar[TrustLevel] = TrustLevel.VERIFIED_EXTERNAL

    def execute(self, state: AgentState) -> dict[str, Any]:
        emit_progress("Checking the request...")
        user_input = state.get("user_input", "")
        input_context = state.get("input_context", {})  # read-only

        if not user_input or not isinstance(user_input, str) or not user_input.strip():
            # Nothing to search, but the caller can simply send a question and
            # try again - so the run completes carrying the reason rather than
            # terminating. A terminal failure here would end the conversation
            # and surface only an exception type, with the reason reachable
            # solely from the audit trail.
            emit_progress(EMPTY_INPUT)
            emit_trace_event(
                "pre_process_rejected",
                {"reason": _CODE_EMPTY_INPUT},
                state,
            )
            return {
                "status": AgentStatus.SUCCESS.value,
                "error_code": _CODE_EMPTY_INPUT,
                "error_log": ["PreProcessNode: user_input is empty or missing"],
            }

        validated_input = _surface_strip_identifiers(user_input.strip())

        # Domain audit: a KB search request was accepted and surface-redacted
        # (no direct identifiers in the payload).
        emit_trace_event(
            "pre_process_complete",
            {"input_chars": len(validated_input)},
            state,
        )

        channel = _inert_label(input_context.get("channel") if isinstance(input_context, dict) else None)

        return {
            "validated_input": validated_input,
            "enriched_context": {
                "source": "AcademicCurriculumKnowledgeAgent",
                "channel": channel,
            },
            "status": AgentStatus.SUCCESS.value,
        }
