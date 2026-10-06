# EDU-C2-005 — Unit Tests: trust gate
#
# The main slot is CurriculumKnowledgeGraphNode in src/graph/graph.py,
# delegating to the 5 inner domain nodes in DomainWorkflowGraph; this suite
# pins the template's full trust matrix (docs/02_design.md).
#
# Trust-gate contract: tests must invoke nodes via node(state) — through
# BaseNode.__call__, which runs the trust gate → PII input mask → execute()
# → output gate — never via node.execute(state) directly, which bypasses the
# gate. A denial RETURNS an error dict (never raises) with status ERROR and
# "trust gate denied" in error_log; execute() never runs, so execute-only
# output keys are ABSENT from the returned dict.
#
# TestAllInnerNodesAdmitAnonymous widens the single InputValidateNode
# ANONYMOUS-admission check to all 5 inner domain nodes, so a future node
# accidentally declaring a stricter required_trust_level fails here instead
# of only being caught by the static TestTrustLevelMatrix class-attribute
# check.

import pytest

from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel

from src.nodes.generate_answer_node import GenerateAnswerNode
from src.nodes.input_validate_node import InputValidateNode
from src.nodes.output_format_node import OutputFormatNode
from src.nodes.post_process_node import PostProcessNode
from src.nodes.pre_process_node import PreProcessNode
from src.nodes.rerank_filter_node import RerankFilterNode
from src.nodes.retrieve_node import RetrieveNode
from src.schemas.state import to_json


def _make_state(
    trust_value: str, user_input: str = "what does the MEXT guideline say about credit-hour requirements?", **extra
) -> dict:
    state = {
        "user_input": user_input,
        "caller_trust_level": trust_value,
        "node_history": [],
        "error_log": [],
        "session_id": "test-session",
        "execution_time": {},
    }
    state.update(extra)
    return state


class TestTrustGate:
    """Trust gate tests — all invocations go through node(state) / __call__."""

    def test_anonymous_caller_allowed_on_inner_node(self):
        """An ANONYMOUS caller passes an ANONYMOUS inner domain node."""
        node = InputValidateNode()  # required_trust_level = ANONYMOUS
        result = node(_make_state(TrustLevel.ANONYMOUS.value, validated_input="credit hour requirements"))
        assert "trust gate denied" not in str(result.get("error_log", []))
        assert result.get("search_query"), "inner node should produce a normalised query"

    def test_anonymous_caller_denied_on_pre_process(self):
        """Trust rejection: ANONYMOUS caller on the VERIFIED_EXTERNAL PreProcessNode.

        __call__ must RETURN an error dict (never raise) with status ERROR and
        'trust gate denied' in the error_log. execute() never ran, so the
        execute-only output key (validated_input) must be ABSENT.
        """
        node = PreProcessNode()  # required_trust_level = VERIFIED_EXTERNAL
        result = node(_make_state(TrustLevel.ANONYMOUS.value))
        assert result.get("status") == AgentStatus.ERROR.value
        error_log = result.get("error_log", [])
        assert any(
            "trust gate denied" in str(e) for e in error_log
        ), f"Expected 'trust gate denied' in error_log, got: {error_log}"
        assert "validated_input" not in result, "execute() must not run on a trust denial — validated_input leaked"

    def test_verified_external_caller_passes_pre_process(self):
        """A VERIFIED_EXTERNAL caller clears the pre_process gate and the
        node writes validated_input."""
        node = PreProcessNode()
        result = node(_make_state(TrustLevel.VERIFIED_EXTERNAL.value))
        assert result.get("status") == AgentStatus.SUCCESS.value
        assert result.get("validated_input")

    def test_pre_process_empty_input_rejected_after_gate(self):
        """The trust gate passes, then the node's own input validation rejects empty input."""
        node = PreProcessNode()
        result = node(_make_state(TrustLevel.VERIFIED_EXTERNAL.value, user_input=""))
        # The gate passes, the node declines the request, and the run completes
        # carrying the reason so the caller can send a question and try again.
        assert result.get("status") == AgentStatus.SUCCESS.value
        assert result.get("error_code") == "EMPTY_INPUT"
        assert not result.get("validated_input")
        assert any("empty" in str(e) for e in result.get("error_log", []))

    def test_anonymous_caller_denied_on_post_process(self):
        """Trust rejection on the other VERIFIED_EXTERNAL outer slot (post_process).

        The denial dict carries no execute-only key (formatted_output ABSENT).
        """
        node = PostProcessNode()  # required_trust_level = VERIFIED_EXTERNAL
        result = node(
            _make_state(
                TrustLevel.ANONYMOUS.value,
                result="a clean processed result",
            )
        )
        assert result.get("status") == AgentStatus.ERROR.value
        assert any("trust gate denied" in str(e) for e in result.get("error_log", []))
        assert "formatted_output" not in result, "execute() must not run on a trust denial — formatted_output leaked"

    def test_verified_external_caller_passes_post_process(self):
        """A VERIFIED_EXTERNAL caller clears the post_process gate."""
        node = PostProcessNode()
        result = node(
            _make_state(
                TrustLevel.VERIFIED_EXTERNAL.value,
                result="a clean processed result",
            )
        )
        assert result.get("status") == AgentStatus.SUCCESS.value
        assert result.get("formatted_output")

    def test_output_gate_blocks_credential_like_output(self):
        """PostProcessNode blocks output carrying a credential-like pattern,
        even from a VERIFIED_EXTERNAL caller (the gate is not suppressible).
        Behavioural assertion: ERROR status + the raw token gone from every
        surfaced field — never the gate's exact wording."""
        node = PostProcessNode()
        secret = "sk-abcdefghijklmnopqrstuvwx"
        result = node(
            _make_state(
                TrustLevel.VERIFIED_EXTERNAL.value,
                result=f"here is a token: {secret}",
            )
        )
        assert result.get("status") == AgentStatus.ERROR.value
        assert result.get("error_log"), "a blocked output must record an error"
        assert secret not in str(result.get("formatted_output", ""))
        assert secret not in str(result.get("result", ""))

    def test_output_gate_recurses_into_nested_structures(self):
        """The output gate is Any-typed and recurses into dict/list — a
        violation nested inside a structured payload must not slip past a
        top-level-only string scan."""
        node = PostProcessNode()
        nested_result = {
            "answer": "a clean grounded answer",
            "citations": [{"passage_id": "kb-001", "note": "Bearer abcdefghijklmnopqrstuvwxyz012345"}],
        }
        result = node(
            _make_state(
                TrustLevel.VERIFIED_EXTERNAL.value,
                result=nested_result,
            )
        )
        assert result.get("status") == AgentStatus.ERROR.value


class TestTrustLevelMatrix:
    """The template's declared trust matrix (docs/02_design.md).

    Outer gate slots (pre_process / post_process) require VERIFIED_EXTERNAL,
    matching the manifest's declared required_trust_level; the five inner
    domain nodes run behind that outer boundary and are declared ANONYMOUS
    per the Cat-2 nested convention.
    """

    def test_outer_gate_nodes_require_verified_external(self):
        assert PreProcessNode.required_trust_level is TrustLevel.VERIFIED_EXTERNAL
        assert PostProcessNode.required_trust_level is TrustLevel.VERIFIED_EXTERNAL

    def test_inner_domain_nodes_admit_anonymous(self):
        for node_cls in (
            InputValidateNode,
            RetrieveNode,
            RerankFilterNode,
            GenerateAnswerNode,
            OutputFormatNode,
        ):
            assert node_cls.required_trust_level is TrustLevel.ANONYMOUS, (
                f"{node_cls.__name__} must declare TrustLevel.ANONYMOUS " "(inner Cat-2 domain node)"
            )


class TestAllInnerNodesAdmitAnonymous:
    """Runtime proof (not just the class-attribute check above) that every
    inner domain node's __call__ actually ADMITS an ANONYMOUS caller — each
    built with the minimal state its own execute() needs so a real trust
    pass/fail is exercised, not a stub."""

    _INNER_NODES_WITH_STATE = (
        (InputValidateNode, {"validated_input": "credit hour requirements"}),
        (RetrieveNode, {"search_query": "credit hour requirements"}),
        (
            RerankFilterNode,
            {
                "retrieved_documents": to_json(
                    [
                        {
                            "id": "kb-001",
                            "title": "t",
                            "category": "mext_course_of_study",
                            "source": "s",
                            "score": 0.9,
                            "excerpt": "e",
                        }
                    ]
                )
            },
        ),
        (GenerateAnswerNode, {"ranked_documents": to_json([]), "search_query": "credit hour requirements"}),
        (OutputFormatNode, {"grounded_answer": "an answer.", "citations": to_json([])}),
    )

    @pytest.mark.parametrize("node_cls,extra_state", _INNER_NODES_WITH_STATE)
    def test_anonymous_caller_admitted(self, node_cls, extra_state):
        state = {
            "caller_trust_level": TrustLevel.ANONYMOUS.value,
            "node_history": [],
            "error_log": [],
            "session_id": "test-session",
            "execution_time": {},
        }
        state.update(extra_state)
        result = node_cls()(state)
        assert "trust gate denied" not in str(
            result.get("error_log", [])
        ), f"{node_cls.__name__} unexpectedly denied an ANONYMOUS caller"
