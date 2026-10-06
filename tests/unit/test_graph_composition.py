# EDU-C2-005 — Unit Tests: nested Cat-2 graph composition (outer + end-to-end)
#
# Drives the REAL outer agent (AcademicCurriculumKnowledgeAgent / Graph)
# end-to-end via AgentBaseGraph.invoke(). The e2e context is
# InvocationContext(caller_trust_level=TrustLevel.VERIFIED_EXTERNAL) — the
# manifest's declared caller level; for_internal() is NEVER used (it would
# over-privilege the run and hide trust-gate regressions).
#
# Mirrors docs/03_test_spec.md §3 (INT-05..INT-13).
# Deterministic — no LLM, no network. framework.* / src.* imports only.


from framework.graph.agent_base_graph import AgentBaseGraph
from framework.schemas.agent_status import AgentStatus
from framework.schemas.invocation_context import InvocationContext
from framework.schemas.trust_level import TrustLevel

from src.graph.domain_workflow_graph import DomainWorkflowGraph
from src.graph.graph import (
    AcademicCurriculumKnowledgeAgent,
    CurriculumKnowledgeGraphNode,
    Graph,
)
from src.nodes.post_process_node import PostProcessNode
from src.nodes.pre_process_node import PreProcessNode
from src.schemas.state import State, from_json, to_json

_MEXT_QUERY = "What does the MEXT guideline say about credit-hour requirements for " "degree programs?"


def _run(user_input: str, trust: TrustLevel = TrustLevel.VERIFIED_EXTERNAL) -> dict:
    ctx = InvocationContext(caller_trust_level=trust, caller_id="unit-suite")
    return Graph().invoke(user_input, ctx=ctx)


class TestOuterGraphConstruction:
    def test_int_05_inherits_agent_base_graph_directly(self):
        assert issubclass(AcademicCurriculumKnowledgeAgent, AgentBaseGraph)

    def test_int_05_graph_alias(self):
        assert Graph is AcademicCurriculumKnowledgeAgent

    def test_state_schema_is_state(self):
        assert AcademicCurriculumKnowledgeAgent().state_schema is State

    def test_int_06_compile_fills_all_backbone_slots(self):
        agent = AcademicCurriculumKnowledgeAgent()
        agent.compile()
        for slot in ("initialize", "pre_process", "main", "post_process", "finalize"):
            assert agent._nodes.get(slot) is not None, f"backbone slot not filled: {slot}"
        assert isinstance(agent._nodes["pre_process"], PreProcessNode)
        assert isinstance(agent._nodes["main"], CurriculumKnowledgeGraphNode)
        assert isinstance(agent._nodes["post_process"], PostProcessNode)

    def test_add_edges_is_not_overridden(self):
        # Backbone wiring belongs to the framework — the template must not
        # redefine it.
        assert "add_edges" not in AcademicCurriculumKnowledgeAgent.__dict__


class TestMainSlotGraphNode:
    def test_int_07_get_subgraph_returns_the_inner_graph(self):
        subgraph = CurriculumKnowledgeGraphNode().get_subgraph()
        assert isinstance(subgraph, DomainWorkflowGraph)
        assert subgraph.config["configurable"]["retrieval"], "inner config must carry the retrieval block"

    def test_int_08_extract_input_prefers_validated_input(self):
        node = CurriculumKnowledgeGraphNode()
        assert node.extract_input({"validated_input": "VI", "user_input": "UI"}) == "VI"
        assert node.extract_input({"user_input": "UI"}) == "UI"

    def test_int_09_merge_output_maps_the_inner_contract(self):
        node = CurriculumKnowledgeGraphNode()
        citations = to_json([{"ref": 1, "id": "kb-001", "title": "t", "source": "s"}])
        delta = node.merge_output(
            {},
            {"formatted_answer": "ANSWER", "citations": citations, "status": AgentStatus.SUCCESS.value},
        )
        # The inner formatted_answer surfaces as BOTH knowledge_base_answer and
        # result (PostProcessNode's output gate reads state["result"]).
        assert delta == {
            "knowledge_base_answer": "ANSWER",
            "result": "ANSWER",
            "citations": citations,
            "status": AgentStatus.SUCCESS.value,
            # Empty when neither layer declined: the key is always present so a
            # reason can never be dropped at the boundary.
            "error_code": "",
        }

    def test_error_strategy_is_propagate_and_hitl_is_contained(self):
        assert CurriculumKnowledgeGraphNode.error_strategy == "propagate"
        assert CurriculumKnowledgeGraphNode.propagate_hitl is False

    def test_int_10_parent_config_never_empty_without_config_file(self):
        # The entry point resolves config/config.yaml and threads it in at
        # register_nodes() time. When nothing reaches the node — an unreadable
        # file yields {} — the forwarded config still carries the fallback
        # retrieval/llm blocks, never {}.
        cfg = CurriculumKnowledgeGraphNode(runtime_config={})._parent_config()
        assert cfg["configurable"]["retrieval"]["kb_path"] == "config/kb/edu_regulatory_kb.json"
        assert cfg["configurable"]["llm"]

    def test_int_13_extract_input_stashes_input_context_for_the_inner_graph(self):
        # GraphNode.execute() does not forward input_context on
        # subgraph.invoke() (SDK 1.0.1) — extract_input() must stash it via
        # the context bridge so DomainWorkflowGraph._extra_initial_state()
        # can seed it into the inner state.
        from src.graph.context_bridge import get_caller_input_context

        node = CurriculumKnowledgeGraphNode()
        node.extract_input({"validated_input": "vi", "input_context": {"category": "institutional_policy"}})
        assert get_caller_input_context() == {"category": "institutional_policy"}
        node.extract_input({"validated_input": "vi"})
        assert get_caller_input_context() == {}


class TestEndToEndInvoke:
    """Full agent run: outer backbone + inner domain workflow, no LLM."""

    def test_int_11_invoke_returns_success(self):
        result = _run(_MEXT_QUERY)
        assert (
            result.get("status") == AgentStatus.SUCCESS.value
        ), f"Expected success, got {result.get('status')}. result={result!r}"

    def test_int_11_output_is_the_gated_formatted_answer(self):
        output = _run(_MEXT_QUERY).get("output")
        assert isinstance(output, str) and output.strip()
        assert output.startswith("# Academic Curriculum Knowledge Base Search Result")
        assert "[1]" in output
        assert "does not constitute an official interpretation of MEXT guidelines" in output

    def test_int_11_e2e_traverses_the_post_process_gate(self):
        history = _run(_MEXT_QUERY).get("node_history", [])
        for cls_name in ("PreProcessNode", "CurriculumKnowledgeGraphNode", "PostProcessNode"):
            assert cls_name in history, f"node_history missing {cls_name}: {history}"

    def test_no_coverage_query_still_terminates_success(self):
        result = _run("quantum telepathy sandwich recipes")
        assert result.get("status") == AgentStatus.SUCCESS.value
        assert "does not contain sufficient coverage" in result.get("output", "")

    def test_int_12_anonymous_caller_is_denied_at_the_outer_boundary(self):
        """Trust gate at graph level: an ANONYMOUS invoke is refused by the
        VERIFIED_EXTERNAL pre_process slot. The error state short-circuits the
        inner domain workflow (the main slot sees status=error and skips
        invoking DomainWorkflowGraph) and routes past post_process to
        finalize — no domain answer is ever produced."""
        result = _run(_MEXT_QUERY, trust=TrustLevel.ANONYMOUS)
        assert result.get("status") == AgentStatus.ERROR.value
        assert not result.get("output")
        history = result.get("node_history", [])
        assert "PostProcessNode" not in history
        assert history[:2] == ["InitializeNode", "PreProcessNode"]


class TestStateRoundTrip:
    """JSON-string State helpers: producers to_json() on write, consumers
    from_json() on read (checkpoint msgpack safety)."""

    def test_to_from_json_list_round_trip(self):
        original = [{"id": "kb-001", "score": 0.7, "title": "credit-hour requirements"}]
        assert from_json(to_json(original)) == original

    def test_to_from_json_dict_round_trip(self):
        original = {"category": "mext_course_of_study", "top_k": 3}
        assert from_json(to_json(original)) == original

    def test_to_json_none_passes_through(self):
        assert to_json(None) is None

    def test_from_json_malformed_returns_default(self):
        assert from_json("{not valid json", default=[]) == []
        assert from_json(None, default={}) == {}
        assert from_json("", default=[]) == []
