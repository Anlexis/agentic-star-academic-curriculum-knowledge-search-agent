# EDU-C2-005 — Unit Tests: PreProcessNode (outer pre_process slot; input
# validation + identifier screen)
#
# Invocation canon: every test invokes the node via node(state) —
# BaseNode.__call__ -> trust gate -> PII input mask -> execute() -> output
# gate — never a bare node.execute(state). PreProcessNode requires
# VERIFIED_EXTERNAL, so its behavioural tests build the state at that level
# (the ANONYMOUS rejection lives in test_trust_gate.py).
#
# Masking layering note: the FRAMEWORK input gate masks user_input /
# validated_input before execute() runs — e-mails, SSN/phone/CC digit groups
# and Title-Case name bigrams surface as [MASKED]. The NODE's own surface
# strip then catches long numeric ID sequences (10-19 digits) the framework
# patterns do not (a 12-digit grouped run IS a framework my_number_jp match;
# an 11-digit ungrouped run is not), and replaces them with [REDACTED].
# Intentional-PII tests therefore assert the raw identifier is GONE and the
# corresponding masked marker is present. All examples below were confirmed
# empirically against the installed real SDK's shared.security.pii_detector
# .detect_pii() and PreProcessNode itself, not hand-derived from the regexes.
#
# Mirrors docs/03_test_spec.md S2.1 (PRE-01..PRE-08).
# Deterministic — no LLM, no network. framework.* / src.* imports only.

from unittest.mock import MagicMock

from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel

import src.nodes.pre_process_node
from src.nodes.pre_process_node import PreProcessNode

# Lowercase EDU regulatory phrasing on purpose: PII-free (no Title-Case
# bigram, no @, no digit run), so the framework mask leaves the payload
# untouched — confirmed empirically (detect_pii() -> []) that "MEXT guideline"
# does NOT match the Title-Case-bigram rule: "MEXT" is all-caps (zero
# lowercase letters) so it fails the per-word [A-Z][a-z]{1,20} pattern.
_VALID_QUERY = "what does the mext guideline say about credit-hour requirements for " "degree programs"


def _make_state(user_input=_VALID_QUERY, **extra) -> dict:
    state = {
        "user_input": user_input,
        "caller_trust_level": TrustLevel.VERIFIED_EXTERNAL.value,
        "node_history": [],
        "error_log": [],
        "session_id": "unit-session",
        "execution_time": {},
    }
    state.update(extra)
    return state


class TestPreProcessSuccess:
    def test_pre_01_valid_query_accepted(self):
        result = PreProcessNode()(_make_state())
        assert result["status"] == AgentStatus.SUCCESS.value
        # Regression guard: State carries the plain string, never the enum.
        assert type(result["status"]) is str  # noqa: E721 — AgentStatus subclasses str; isinstance() would not catch the bare enum
        assert result["validated_input"] == _VALID_QUERY

    def test_enriched_context_carries_channel(self):
        result = PreProcessNode()(_make_state(input_context={"channel": "web"}))
        assert result["enriched_context"]["channel"] == "web"
        assert result["enriched_context"]["source"] == "AcademicCurriculumKnowledgeAgent"

    def test_missing_channel_defaults_to_unknown(self):
        result = PreProcessNode()(_make_state())
        assert result["enriched_context"]["channel"] == "unknown"

    def test_free_text_channel_is_locked_to_unknown(self):
        # A caller-supplied telemetry label is locked to an inert identifier
        # shape ([a-z0-9_]{1,32}) before it is written to state — free text
        # there is a log/output injection vector. The rejected value is
        # replaced, never echoed.
        for hostile in ("Web Portal!", "<script>x</script>", "A" * 40, 123, {"x": 1}):
            result = PreProcessNode()(_make_state(input_context={"channel": hostile}))
            assert result["enriched_context"]["channel"] == "unknown"


class TestPreProcessRejection:
    def test_pre_02_empty_input_is_error(self):
        result = PreProcessNode()(_make_state(user_input=""))
        assert result["status"] == AgentStatus.SUCCESS.value
        # The run completes and carries the reason: the caller
        # can correct the value and send the request again.
        assert result["error_code"] == "EMPTY_INPUT"
        assert result["error_log"]
        # No validated_input is produced on the reject path.
        assert "validated_input" not in result

    def test_whitespace_only_is_error(self):
        result = PreProcessNode()(_make_state(user_input="   \n\t "))
        assert result["status"] == AgentStatus.SUCCESS.value
        # The run completes and carries the reason: the caller
        # can correct the value and send the request again.
        assert result["error_code"] == "EMPTY_INPUT"

    def test_pre_03_missing_user_input_is_error(self):
        state = _make_state()
        del state["user_input"]
        result = PreProcessNode()(state)
        assert result["status"] == AgentStatus.SUCCESS.value
        # The run completes and carries the reason: the caller
        # can correct the value and send the request again.
        assert result["error_code"] == "EMPTY_INPUT"

    def test_non_string_input_is_error(self):
        result = PreProcessNode()(_make_state(user_input={"malicious": "dict"}))
        assert result["status"] == AgentStatus.SUCCESS.value
        # The run completes and carries the reason: the caller
        # can correct the value and send the request again.
        assert result["error_code"] == "EMPTY_INPUT"


class TestPreProcessIdentifierScreen:
    """PRE-04: raw identifiers never survive into validated_input.

    Every raw string / expected marker pair below was run through the real
    PreProcessNode (and shared.security.pii_detector.detect_pii directly) in
    the installed real SDK before being pinned here.
    """

    def test_long_student_id_redacted_by_node_screen(self):
        # An 11-digit ungrouped run is NOT a framework pattern (not 12-digit
        # my_number_jp, not 16-digit credit_card) — only the node's own
        # 10-19-digit screen catches it -> [REDACTED], never [MASKED].
        raw = "verify course credit records for student id 20231234567 before certification"
        result = PreProcessNode()(_make_state(user_input=raw))
        vi = result["validated_input"]
        assert result["status"] == AgentStatus.SUCCESS.value
        assert "20231234567" not in vi
        assert "[REDACTED]" in vi
        assert "[MASKED]" not in vi

    def test_email_masked_by_framework_s2_gate(self):
        raw = "escalate the accreditation dossier to registrar.office@example.edu today"
        result = PreProcessNode()(_make_state(user_input=raw))
        vi = result["validated_input"]
        assert "registrar.office@example.edu" not in vi
        assert "[MASKED]" in vi

    def test_grouped_record_digits_masked(self):
        # 4-4-4 digit groups match the framework my_number_jp pattern.
        raw = "record 1234 5678 9012 shows a pending transcript hold"
        result = PreProcessNode()(_make_state(user_input=raw))
        vi = result["validated_input"]
        assert "1234 5678 9012" not in vi
        assert "[MASKED]" in vi

    def test_title_case_bigram_masked(self):
        # "Curriculum Committee": both words are [A-Z][a-z]+ -> framework
        # name-bigram pattern masks it.
        raw = "escalate the Curriculum Committee review to the registrar"
        result = PreProcessNode()(_make_state(user_input=raw))
        vi = result["validated_input"]
        assert "Curriculum Committee" not in vi
        assert "[MASKED]" in vi


class TestPreProcessAudit:
    def test_pre_08_domain_audit_payload(self, monkeypatch):
        """Audit: the accepted request emits pre_process_complete; the assertion
        targets call.args[1] — the event payload — never the whole call repr."""
        spy = MagicMock()
        monkeypatch.setattr(src.nodes.pre_process_node, "emit_trace_event", spy)
        result = PreProcessNode()(_make_state())
        events = [call.args[0] for call in spy.call_args_list]
        assert "pre_process_complete" in events
        payload = spy.call_args_list[events.index("pre_process_complete")].args[1]
        assert payload["input_chars"] == len(result["validated_input"])
