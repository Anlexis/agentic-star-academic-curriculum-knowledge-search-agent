"""AgentCore Platform v1.0"""

# EDU-C2-005 - InputValidateNode
# Domain node 1: the caller-data contract gate. Parses and normalises the
# incoming KB search request and validates every caller-controlled parameter
# against explicit bounds - fail-closed.
#
# Caller parameters arrive on two channels, validated with the SAME rules:
#
#   1. input_context (structured invocation parameters, bridged from the
#      outer graph - see src/graph/context_bridge.py):
#          {"category": "<slug>", "top_k": <int>}
#   2. a JSON envelope inside the request string (legacy in-band form):
#          {"query": "...", "category": "...", "top_k": N}
#
# When both channels supply the same field, input_context (the structured,
# first-class channel) takes precedence - but an INVALID value on either
# channel rejects the request outright; it never falls back to the other
# channel's value.
#
# Validation contract (fail CLOSED, field-naming errors only):
#   - top_k: integer (bool explicitly rejected - isinstance(True, int) is
#     True in Python), within 1..20. Non-integers, numeric strings,
#     non-finite floats (NaN / Infinity parse fine via float() and Python's
#     json even accepts bare NaN in request bodies) and out-of-range values
#     all reject the request with ERROR status naming the FIELD - never
#     echoing the value.
#   - category: one of the four fixed KB category slugs. The slugs are inert
#     identifiers ([a-z0-9_]{1,32}); anything else rejects the request. The
#     rejected value is never echoed into errors, notes, or output.
#   - absent fields are fine: the search simply runs with defaults
#     (unfiltered, configured top_k).
#   - query: refused when it carries a prompt-injection instruction
#     (instruction-override phrasing aimed at the model rather than the
#     knowledge base). The platform input gate refuses these too, but the
#     template MUST NOT depend on that: where the gate is absent or
#     configured off, an unchecked payload would reach the answer path and
#     return success. Refusal is stated in terms of BEHAVIOUR (error, no
#     answer, no citations), never a gate's wording.
#
# Wired by the inner graph (DomainWorkflowGraph). On rejection every
# downstream node is skipped by the framework's ERROR short-circuit, so no
# retrieval or answer assembly runs on unvalidated data.
# Returns only changed state keys (partial dict).

import json
import re
from typing import Any, ClassVar, Dict, List, Optional, Tuple

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_state import AgentState
from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel
from shared.utils.audit_logger import emit_trace_event
from src.services.progress import emit_progress
from src.services.failure_message import INVALID_VALUE
from src.schemas.state import to_json

# Marker written to state when the run completes without an answer because a
# caller-supplied value could not be accepted. The caller can correct the value
# and send the request again, so the run completes and carries the reason
# instead of terminating - a terminal failure would close the conversation and
# report only an exception type.
_CODE_INVALID_REQUEST = "INVALID_REQUEST"

# Audit reason for content this node refuses outright. Kept distinct from the
# marker above so a refusal is never presented to the caller as something a
# corrected request would get past.
_REASON_REFUSED = "REQUEST_REFUSED"

# Hard cap on the normalised query length (defence-in-depth on input size).
_MAX_QUERY_CHARS = 2000

# Bounds for the caller-supplied top_k override (untrusted numeric guard).
_TOP_K_MIN = 1
_TOP_K_MAX = 20

# Instruction-override phrasings: text addressed to the MODEL rather than a
# question addressed to the knowledge base. Deliberately narrow — a genuine
# question about these words ("what does 'system prompt' mean?") does not match,
# because each alternative requires the imperative override shape.
_INJECTION_RE = re.compile(
    r"ignore\s+(?:all\s+|any\s+)?(?:previous|prior|above|earlier)\s+"
    r"(?:instruction|instructions|prompt|prompts|rule|rules|direction|directions)"
    r"|disregard\s+(?:all\s+|any\s+)?(?:previous|prior|above|earlier)\s+"
    r"(?:instruction|instructions|prompt|prompts|rule|rules)"
    r"|forget\s+(?:all\s+|any\s+)?(?:previous|prior|above|earlier)\s+"
    r"(?:instruction|instructions|prompt|prompts)"
    r"|(?:reveal|show|print|repeat|output|disclose)\s+(?:me\s+)?(?:your|the)\s+"
    r"(?:system\s+prompt|system\s+message|instructions|initial\s+prompt)"
    r"|you\s+are\s+now\s+(?:a|an)\s"
    r"|act\s+as\s+(?:if\s+you\s+are\s+)?(?:a\s+|an\s+)?(?:developer|admin|root)\s+mode"
    r"|override\s+(?:your|the)\s+(?:instruction|instructions|rules|safety)",
    re.IGNORECASE,
)

_WHITESPACE_RE = re.compile(r"\s+")

# Caller strings that select behaviour must be inert identifiers - lowercase
# alphanumerics/underscore, bounded length. Free text in such a field is
# caller-controlled output/log injection and is rejected before comparison.
_INERT_IDENTIFIER_RE = re.compile(r"^[a-z0-9_]{1,32}$")

# The four fixed KB categories this template's corpus is organised around
# (docs/02_design.md). Every slug satisfies _INERT_IDENTIFIER_RE.
VALID_CATEGORIES: List[str] = [
    "mext_course_of_study",
    "niad_qe_accreditation",
    "curriculum_requirements",
    "institutional_policy",
]


def _validate_top_k(value: Any) -> Tuple[Optional[int], Optional[str]]:
    """Validate the untrusted caller top_k. Returns (value, error).

    Fail-closed: only a plain integer within [1, 20] is accepted. Booleans,
    floats (including NaN/Infinity, which parse via float() and arrive intact
    through raw JSON), numeric strings, and out-of-range integers all return
    a field-naming error - the offending VALUE is never echoed.
    """
    if value is None:
        return None, None
    if isinstance(value, bool) or not isinstance(value, int):
        return None, (f"InputValidateNode: top_k must be an integer between " f"{_TOP_K_MIN} and {_TOP_K_MAX}.")
    if not _TOP_K_MIN <= value <= _TOP_K_MAX:
        return None, (f"InputValidateNode: top_k must be an integer between " f"{_TOP_K_MIN} and {_TOP_K_MAX}.")
    return value, None


def _validate_category(value: Any) -> Tuple[Optional[str], Optional[str]]:
    """Validate the untrusted caller category filter. Returns (value, error).

    Fail-closed: only one of the four fixed category slugs (after
    lowercase/strip normalisation) is accepted. The candidate must already be
    an inert identifier before it is even compared against the fixed set, and
    a rejected value is never echoed into the error.
    """
    if value is None:
        return None, None
    if not isinstance(value, str):
        return None, ("InputValidateNode: category must be one of the fixed category slugs.")
    candidate = value.strip().lower()
    if not candidate:
        return None, None
    if not _INERT_IDENTIFIER_RE.match(candidate) or candidate not in VALID_CATEGORIES:
        return None, ("InputValidateNode: category must be one of the fixed category slugs.")
    return candidate, None


class InputValidateNode(FunctionNode):
    """Parse the request into a normalised query + validated filters.

    Input state keys:
        validated_input | user_input: identifier-stripped request payload
        input_context:                structured invocation parameters
                                      (category / top_k), bridged from the
                                      outer graph

    Output state keys (partial dict):
        search_query:  normalised free-text search query
        query_filters: JSON dict {"category": str|None, "top_k": int|None}
        intake_notes:  (when non-fatal anomalies were seen) JSON list[str]

    On a caller-contract violation:
        status:    ERROR (fail-closed; downstream nodes are skipped)
        error_log: one field-naming message per violation - values are
                   never echoed
    """

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.ANONYMOUS

    def execute(self, state: AgentState) -> dict[str, Any]:  # noqa: C901
        emit_progress("Checking the request...")
        raw = state.get("validated_input") or state.get("user_input", "")
        input_context = state.get("input_context", {}) or {}
        notes: List[str] = []
        # Two rejection classes with different reporting, never the same list:
        #   refusals - content this node refuses to process at all. Terminal.
        #   errors   - a caller-supplied value that could not be accepted.
        #              The caller can correct it, so the run completes with a
        #              marker instead of terminating.
        refusals: List[str] = []
        errors: List[str] = []

        # ------------------------------------------------------------------
        # Channel 1: the request string (plain text or JSON envelope)
        # ------------------------------------------------------------------
        query = ""
        envelope_category: Any = None
        envelope_top_k: Any = None

        if isinstance(raw, str) and raw.strip():
            payload: Any = None
            text = raw.strip()
            if text.startswith("{"):
                try:
                    payload = json.loads(text)
                except (json.JSONDecodeError, ValueError):
                    notes.append(
                        "InputValidateNode: JSON-looking input did not parse - " "treated as plain text query."
                    )
            if isinstance(payload, dict):
                query = str(payload.get("query") or payload.get("question") or "")
                envelope_category = payload.get("category")
                envelope_top_k = payload.get("top_k")
            else:
                query = text
        else:
            notes.append("InputValidateNode: empty request - no query to search.")

        # Instruction-override payloads are refused here, before any
        # retrieval or answer assembly, independently of the platform gate.
        if query and _INJECTION_RE.search(query):
            refusals.append("InputValidateNode: query refused - instruction-override content.")

        # ------------------------------------------------------------------
        # Channel 2: structured invocation parameters (input_context)
        # ------------------------------------------------------------------
        if not isinstance(input_context, dict):
            # A non-mapping input_context cannot carry a valid contract.
            errors.append("InputValidateNode: input_context must be an object.")
            input_context = {}
        context_category = input_context.get("category")
        context_top_k = input_context.get("top_k")

        # ------------------------------------------------------------------
        # Validate both channels with the same rules; input_context wins on
        # overlap; an invalid value on EITHER channel rejects the request.
        # ------------------------------------------------------------------
        category: Optional[str] = None
        top_k: Optional[int] = None

        for source_value in (envelope_category, context_category):
            category_value, category_err = _validate_category(source_value)
            if category_err:
                errors.append(category_err)
            elif category_value is not None:
                category = category_value  # context (validated last) wins on overlap

        for source_value in (envelope_top_k, context_top_k):
            top_k_value, top_k_err = _validate_top_k(source_value)
            if top_k_err:
                errors.append(top_k_err)
            elif top_k_value is not None:
                top_k = top_k_value  # context (validated last) wins on overlap

        # Refused content terminates the run. The request is not something the
        # caller should re-send in corrected form, so it is not reported as a
        # completion.
        if refusals:
            emit_progress(INVALID_VALUE)
            emit_trace_event(
                "input_validate_refused",
                {"reason": _REASON_REFUSED, "refusal_count": len(refusals)},
                state,
            )
            return {
                "status": AgentStatus.ERROR.value,
                "error_log": list(dict.fromkeys(refusals)),
            }

        if errors:
            # Fail CLOSED on the retrieval path: no search runs on a contract
            # this node could not accept. The run still COMPLETES, carrying the
            # reason, so the caller can correct the value and send it again on
            # the same conversation. Deduplicate while preserving order (both
            # channels may fail the same field the same way).
            deduped = list(dict.fromkeys(errors))
            emit_progress(INVALID_VALUE)
            emit_trace_event(
                "input_validate_rejected",
                {"reason": _CODE_INVALID_REQUEST, "error_count": len(deduped)},
                state,
            )
            return {
                "status": AgentStatus.SUCCESS.value,
                "error_code": _CODE_INVALID_REQUEST,
                "error_log": deduped,
                "intake_notes": to_json(notes),
            }

        # Normalise whitespace and cap length.
        query = _WHITESPACE_RE.sub(" ", query).strip()
        if len(query) > _MAX_QUERY_CHARS:
            query = query[:_MAX_QUERY_CHARS]
            notes.append(f"InputValidateNode: query truncated to {_MAX_QUERY_CHARS} chars.")

        filters = {"category": category, "top_k": top_k}

        # Domain audit: request parsed, validated, and normalised.
        emit_trace_event(
            "input_validate_complete",
            {
                "query_chars": len(query),
                "has_category_filter": category is not None,
                "has_top_k_override": top_k is not None,
            },
            state,
        )

        out: Dict[str, Any] = {
            "search_query": query,
            "query_filters": to_json(filters),
        }
        if notes:
            out["intake_notes"] = to_json(notes)
        return out
