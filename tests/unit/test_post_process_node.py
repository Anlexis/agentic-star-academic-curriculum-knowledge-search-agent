# EDU-C2-005 — Unit Tests: PostProcessNode (outer post_process slot; output gate)
#
# Invocation canon: node(state) via BaseNode.__call__. PostProcessNode is the
# second outer gate slot and requires VERIFIED_EXTERNAL (like PreProcessNode),
# so its behavioural tests build the state at that level; the ANONYMOUS
# rejection lives in test_trust_gate.py.
#
# Gate layering: the node's own module-level _security_gate_output() scan runs
# INSIDE execute() and replaces a violating answer with the sanitised stub
# (returned dict — no exception). The framework's FunctionNode credential
# scan then sees only the clean stub. Intentional-credential tests assert the
# raw secret never survives into formatted_output OR result.
#
# Mirrors docs/03_test_spec.md §2.7 (POST-01..POST-06).
# Deterministic — no LLM, no network. framework.* / src.* imports only.

from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel

from src.nodes.post_process_node import PostProcessNode

_CLEAN_REPORT = (
    "# Academic Curriculum Knowledge Base Search Result\n\n"
    "[1] minimum credit-hour requirements are set by the course-of-study guideline.\n"
)

# JWT-shaped token built at runtime so no credential-shaped literal ever sits
# in the repository (credential-scan hygiene).
_FAKE_JWT = "eyJ" + "a" * 12 + "." + "b" * 12 + "." + "c" * 12


def _make_state(result_text, **extra) -> dict:
    state = {
        "result": result_text,
        "caller_trust_level": TrustLevel.VERIFIED_EXTERNAL.value,
        "node_history": [],
        "error_log": [],
        "session_id": "unit-session",
        "execution_time": {},
    }
    state.update(extra)
    return state


class TestPostProcessClean:
    def test_post_01_clean_output_passes_through(self):
        result = PostProcessNode()(_make_state(_CLEAN_REPORT))
        assert result["status"] == AgentStatus.SUCCESS.value
        # Regression guard: State carries the plain string, never the enum.
        assert type(result["status"]) is str  # noqa: E721 — AgentStatus subclasses str; isinstance() would not catch the bare enum
        # The clean answer passes through with the standing advisory
        # disclaimer enforced (already-composed answers are unchanged; see
        # TestDisclaimerEnforcement for the repair path).
        assert result["formatted_output"].startswith(_CLEAN_REPORT)

    def test_post_02_empty_result_is_non_fatal(self):
        result = PostProcessNode()(_make_state(""))
        assert result["status"] == AgentStatus.SUCCESS.value
        assert result["formatted_output"] == ""


class TestPostProcessOutputGate:
    def _assert_blocked(self, result, secret):
        assert result["status"] == AgentStatus.ERROR.value
        assert any("output blocked" in str(e) for e in result["error_log"])
        # The raw secret must not survive into either surfaced field.
        assert secret not in str(result.get("formatted_output", ""))
        assert secret not in str(result.get("result", ""))
        assert "[OUTPUT BLOCKED by the output security gate" in result["formatted_output"]

    def test_post_03_api_key_is_blocked(self):
        secret = "sk-ABCDEF0123456789abcdef"
        result = PostProcessNode()(_make_state(f"# Report\n\n<!-- debug api_key={secret} -->\n"))
        self._assert_blocked(result, secret)

    def test_post_04_credential_assignment_is_blocked(self):
        secret = "password=super_secret_value_123"
        result = PostProcessNode()(_make_state(f"# Report\n\ninternal note: {secret}\n"))
        self._assert_blocked(result, "super_secret_value_123")

    def test_post_05_jwt_is_blocked(self):
        result = PostProcessNode()(_make_state(f"# Report\n\nsession token {_FAKE_JWT}\n"))
        self._assert_blocked(result, _FAKE_JWT)

    def test_post_06_bearer_token_is_blocked(self):
        secret = "Bearer abcdefghijklmnopqrstuvwxyz0123456789"
        result = PostProcessNode()(_make_state(f"# Report\n\nauthorization: {secret}\n"))
        self._assert_blocked(result, secret)


class TestPostProcessNestedScan:
    """Deeper nested-structure coverage than the seed suite's 2-level case
    (tests/unit/test_trust_gate.py::test_output_gate_recurses_into_nested_structures)
    — this class exercises a clean nested payload and a violation buried a
    level deeper (dict -> list -> dict), never duplicating that seed test."""

    def test_clean_nested_structure_passes_through(self):
        # A structured (non-string) result with no credential-shaped value
        # anywhere must NOT be blocked — the recursive scan is precise, not
        # merely "any non-string result is suspicious".
        clean_result = {
            "answer": "a clean grounded answer",
            "citations": [{"id": "kb-001", "note": "no sensitive content here"}],
        }
        result = PostProcessNode()(_make_state(clean_result))
        assert result["status"] == AgentStatus.SUCCESS.value
        assert result["formatted_output"] == clean_result

    def test_output_gate_recurses_three_levels_deep(self):
        # dict -> list -> dict: one level deeper than the seed suite's
        # dict -> list case. The module-level gate is Any-typed and recurses
        # dict/list/tuple/set, so a violation must still be caught however
        # deeply it is buried - a top-level-only string scan is not enough.
        nested_result = {
            "answer": "a clean grounded answer",
            "sources": [
                {
                    "id": "kb-001",
                    "meta": {"debug_note": "api_key=sk-ABCDEF0123456789abcdef"},
                }
            ],
        }
        result = PostProcessNode()(_make_state(nested_result))
        assert result["status"] == AgentStatus.ERROR.value
        assert "[OUTPUT BLOCKED by the output security gate" in result["formatted_output"]


class TestDisclaimerEnforcement:
    """The documented output contract says EVERY answer carries the standing
    advisory disclaimer. Composition (OutputFormatNode) guarantees it today;
    this gate independently ENFORCES it so a future rendering change cannot
    silently drop the invariant."""

    def test_answer_with_disclaimer_is_unchanged(self):
        from src.nodes.output_format_node import _ADVISORY_DISCLAIMER

        composed = f"{_CLEAN_REPORT}\n---\n\n*{_ADVISORY_DISCLAIMER}*"
        result = PostProcessNode()(_make_state(composed))
        assert result["status"] == AgentStatus.SUCCESS.value
        assert result["formatted_output"] == composed
        assert result["formatted_output"].count(_ADVISORY_DISCLAIMER) == 1

    def test_missing_disclaimer_is_repaired_with_audit_event(self, monkeypatch):
        from unittest.mock import MagicMock

        import src.nodes.post_process_node as ppn
        from src.nodes.output_format_node import _ADVISORY_DISCLAIMER

        spy = MagicMock()
        monkeypatch.setattr(ppn, "emit_trace_event", spy)
        result = PostProcessNode()(_make_state(_CLEAN_REPORT))
        assert result["status"] == AgentStatus.SUCCESS.value
        assert _ADVISORY_DISCLAIMER in result["formatted_output"]
        assert _ADVISORY_DISCLAIMER in result["result"]
        events = [call.args[0] for call in spy.call_args_list]
        assert "output_disclaimer_enforced" in events

    def test_blocked_output_is_not_disclaimer_stamped(self):
        # The enforcement applies to CLEAN answers only — the blocked stub
        # replaces the output entirely and stays as-is.
        from src.nodes.output_format_node import _ADVISORY_DISCLAIMER

        secret = "sk-ABCDEF0123456789abcdef"
        result = PostProcessNode()(_make_state(f"api_key={secret}"))
        assert result["status"] == AgentStatus.ERROR.value
        assert _ADVISORY_DISCLAIMER not in result["formatted_output"]
