"""AgentCore Platform v1.0"""

# EDU-C2-005 - Outer graph (AgentBaseGraph; Cat 2 two-layer nested architecture)
#
# Academic Curriculum Knowledge Agent (Cat 2 RAG domain workflow).
#
# Architecture (Cat 2):
#
#   Outer backbone (fixed - identical to Cat 1, do NOT override add_edges()):
#     START -> initialize -> pre_process -> main -> {route} -> post_process -> finalize -> END
#                                             |  (RETRY, max 3)
#                                             -> pre_process
#
#   `main` slot is a GraphNode subclass (CurriculumKnowledgeGraphNode) that
#   delegates the full curriculum-knowledge domain workflow to
#   DomainWorkflowGraph (inner BaseGraph: input_validate -> retrieve ->
#   rerank_filter -> generate_answer -> output_format).
#
#   Domain complexity is fully encapsulated inside the inner graph. The outer
#   backbone is never modified.
#
# Directory layout:
#   src/graph/graph.py                 <- outer graph (this file)
#   src/graph/domain_workflow_graph.py <- inner graph (multi-step topology)
#   src/graph/context_bridge.py        <- input_context hand-off outer -> inner
#
# Class-name contract:
#   graph.py class:           AcademicCurriculumKnowledgeAgent (this file)
#   config/agent.yaml class:  "src.graph.graph.AcademicCurriculumKnowledgeAgent"
#   src/api/server.py import: from src.graph.graph import AcademicCurriculumKnowledgeAgent
#
# Rules enforced:
#   - AcademicCurriculumKnowledgeAgent inherits AgentBaseGraph (framework base
#     class - direct framework inheritance)
#   - super().register_nodes() called first (fills initialize + finalize)
#   - CurriculumKnowledgeGraphNode assigned to self._nodes["main"]
#   - _parent_config() forwards the config/config.yaml retrieval/llm blocks
#     (never {})
#   - extract_input() bridges the caller's input_context to the inner graph
#     (src/graph/context_bridge.py)
#   - merge_output() returns only changed keys
#   - get_output() EXTENDS super().get_output() - structured `citations`
#     surfaced ONLY on SUCCESS, and re-scanned by the output security gate
#     (fail-closed)
#   - add_edges() NOT overridden on the outer graph
#   - No platform-internal SDK imports

from typing import TYPE_CHECKING, Any, ClassVar, Dict, cast

from framework.graph.agent_base_graph import AgentBaseGraph
from framework.nodes.graph_node import GraphNode
from framework.schemas.agent_state import AgentState
from framework.schemas.agent_status import AgentStatus
from src.graph.context_bridge import set_caller_input_context
from src.nodes.post_process_node import PostProcessNode, _security_gate_output
from src.nodes.pre_process_node import PreProcessNode
from src.schemas.state import State, from_json

if TYPE_CHECKING:
    from src.graph.domain_workflow_graph import DomainWorkflowGraph

# Fallbacks mirror the `retrieval` / `llm` blocks in config/config.yaml so
# _parent_config() never forwards an empty config even if the file is
# unreadable in an exotic deployment layout.
_FALLBACK_RETRIEVAL: dict[str, Any] = {
    "top_k": 4,
    "score_threshold": 0.25,
    "kb_path": "config/kb/edu_regulatory_kb.json",
}
_FALLBACK_LLM: dict[str, Any] = {
    "temperature": 0.0,
    "max_tokens": 1500,
}


class CurriculumKnowledgeGraphNode(GraphNode):
    """GraphNode subclass assigned to the `main` slot of the outer agent.

    Wraps DomainWorkflowGraph (inner Cat 2 BaseGraph RAG pipeline).
    Called by the AgentBaseGraph backbone after pre_process and before
    post_process.

    Contracts:
      get_subgraph()    - instantiate DomainWorkflowGraph with the forwarded
                          runtime config (_parent_config())
      extract_input()   - pull validated_input (identifier-stripped) from the
                          outer state and stash input_context for the inner
                          graph (context bridge)
      merge_output()    - map sub_result fields into the outer state delta
                          (changed keys only)
      error_strategy    - "propagate": re-raise inner errors as SubgraphError
                          (fail-fast; the framework converts the raised error
                          into a terminal ERROR status on the outer state)
    """

    def __init__(self, runtime_config: dict[str, Any] | None = None) -> None:
        """Receive the runtime config from the outer graph.

        A BaseNode has no config back-reference of its own, so the outer
        AgentBaseGraph reads `self.config` and threads it in here at
        register_nodes() time. Static construction input - not mutable state.
        """
        self._runtime_config: dict[str, Any] = dict(runtime_config or {})

    # "propagate": re-raise inner graph exceptions as SubgraphError (default - fail fast).
    # "handle": call on_subgraph_error() instead - use for graceful degradation.
    error_strategy: ClassVar[str] = "propagate"

    # False: this template does not use human-in-the-loop interrupts at all.
    propagate_hitl: ClassVar[bool] = False

    def _parent_config(self) -> dict[str, Any]:
        """Forward the `retrieval` + `llm` tuning blocks to the inner graph.

        Loads config/config.yaml (the runtime parameters file) and returns the
        two tuning blocks under config["configurable"] - never an empty dict.
        The inner graph republishes the `retrieval` block into inner state
        (DomainWorkflowGraph._extra_initial_state()) so RetrieveNode /
        RerankFilterNode read live top_k / score_threshold values instead of
        dead configuration text. The `llm` block is forwarded verbatim for
        the documented LLM-synthesis upgrade (unused by the current
        deterministic nodes).
        """
        runtime = self._runtime_config
        retrieval = runtime.get("retrieval")
        if not isinstance(retrieval, dict) or not retrieval:
            retrieval = dict(_FALLBACK_RETRIEVAL)
        llm = runtime.get("llm")
        if not isinstance(llm, dict) or not llm:
            llm = dict(_FALLBACK_LLM)
        return {"configurable": {"retrieval": retrieval, "llm": llm}}

    def get_subgraph(self) -> "DomainWorkflowGraph":
        """Instantiate and return the inner domain workflow graph.

        DomainWorkflowGraph is imported lazily (inside the method) to avoid
        circular-import risk at module load time.

        The inner graph receives the runtime config via its BaseGraph ctor
        (graph-level constructor injection of immutable config - distinct
        from the per-node execute() contract); its domain NODES still take no
        constructor arguments and read config exclusively via the
        state-seeded `retrieval_config` field (execute(self, state) -> dict,
        no config parameter).
        """
        from src.graph.domain_workflow_graph import DomainWorkflowGraph

        return DomainWorkflowGraph(config=self._parent_config())

    def execute(self, state: AgentState) -> dict[str, Any]:
        """Skip the inner graph when the request was already found unacceptable.

        A request rejected by pre_process has no validated_input to search on,
        so running the inner graph would only produce a second, vaguer reason
        for the same rejection - and overwrite the specific one already
        settled. Completing here keeps the original reason intact.

        This override is deliberate: GraphNode.execute() is not final, and the
        marker is the only signal that distinguishes "nothing to do" from "not
        run yet".
        """
        marker = state.get("error_code")
        if marker:
            return {"status": AgentStatus.SUCCESS.value, "error_code": marker}
        return cast(Dict[str, Any], super().execute(state))

    def extract_input(self, state: AgentState) -> str:
        """Return the string input passed into inner_graph.invoke().

        PreProcessNode validates and identifier-strips the raw user_input and
        writes the result to validated_input. Prefer that; fall back to
        user_input if validated_input is absent (e.g. in unit tests).

        Also bridges the caller's input_context to the inner graph:
        GraphNode.execute() does not forward input_context on subgraph.invoke()
        (SDK 1.0.1), and extract_input is the last template-code hook that sees
        the outer state before the inner invoke - see
        src/graph/context_bridge.py. The bridged values are UNVALIDATED here;
        the inner InputValidateNode enforces the caller-data contract
        (fail-closed) before any downstream node reads them.
        """
        set_caller_input_context(state.get("input_context") or {})
        return cast(str, state.get("validated_input") or state.get("user_input", ""))

    def merge_output(self, state: AgentState, sub_result: dict[str, Any]) -> dict[str, Any]:
        """Map inner graph sub_result back into the outer state delta.

        sub_result is the dict returned by DomainWorkflowGraph.get_output().
        Returns ONLY changed keys - never the full state.

        GraphNode.execute() raises SubgraphError before calling merge_output
        when the inner graph terminates with ERROR status (error_strategy
        "propagate"), so this method only runs on a successful inner pass.

        Key coupling (designed together with DomainWorkflowGraph.get_output()):
          Inner get_output() emits  -> "formatted_answer", "citations", "status", ...
          This merge_output() reads -> sub_result.get("formatted_answer"),
                                       sub_result.get("citations"),
                                       sub_result.get("status")

        knowledge_base_answer (str | None): final rendered KB answer; written
          by OutputFormatNode inside the inner graph.
        result: PostProcessNode (outer post_process slot) reads
          state.get("result") - the inner graph emits the rendered answer
          under "formatted_answer", so map it to "result" as well; otherwise
          the final output surfaced by PostProcessNode (and the output
          security gate) is always empty.
        citations: re-surfaced at the outer layer (still a JSON string - the
          State msgpack-safety convention) so
          AcademicCurriculumKnowledgeAgent.get_output() can read it without
          reaching into the inner graph.
        status (str | None): terminal status value from the inner graph run.
        """
        return {
            "knowledge_base_answer": sub_result.get("formatted_answer"),
            "result": sub_result.get("formatted_answer"),
            "citations": sub_result.get("citations"),
            "status": sub_result.get("status"),
            # Outer reason wins. A reason already settled before the inner run
            # is the real one; the inner graph only ever sees the downstream
            # consequence of it ("there was no query to search"), so taking the
            # inner value first would replace a specific reason with a generic
            # one - and a plain sub_result.get() would erase the outer reason
            # entirely whenever the inner run did not set its own.
            "error_code": state.get("error_code") or sub_result.get("error_code", ""),
        }


class AcademicCurriculumKnowledgeAgent(AgentBaseGraph):
    """Outer graph for EDU-C2-005 (Cat 2 RAG, nested).

    Inherits AgentBaseGraph (framework base class) directly. Domain logic is
    fully encapsulated in CurriculumKnowledgeGraphNode (main slot), which
    delegates to DomainWorkflowGraph (inner BaseGraph).

    Backbone (fixed - identical to Cat 1):
        START -> initialize -> pre_process -> main -> post_process -> finalize -> END

    register_nodes() and get_output() are the ONLY overrides:
      - super().register_nodes() fills: initialize, finalize (framework defaults)
      - pre_process:  PreProcessNode (input validation + identifier screen)
      - main:         CurriculumKnowledgeGraphNode (delegates to DomainWorkflowGraph)
      - post_process: PostProcessNode (output security gate + schema enforcement)
      - get_output(): extends the base envelope with the structured
        `citations` field, ONLY on SUCCESS, re-scanned fail-closed

    add_edges() is NOT overridden - backbone wiring belongs to the framework.
    """

    @property
    def name(self) -> str:
        """Agent identifier registered with AgentRegistry."""
        return "AcademicCurriculumKnowledgeAgent"

    @property
    def state_schema(self) -> type:
        return State

    def register_nodes(self) -> None:
        """Fill all 5 backbone slots.

        super().register_nodes() MUST be called first - it injects the
        framework's default InitializeNode (sets schema_version, session_id,
        trust_level) and FinalizeNode (builds response_metadata, total_time_ms).
        """
        super().register_nodes()  # fills: initialize, finalize

        self._nodes["pre_process"] = PreProcessNode()
        self._nodes["main"] = CurriculumKnowledgeGraphNode(runtime_config=self.config)
        self._nodes["post_process"] = PostProcessNode()

    # add_edges() is NOT overridden - backbone wiring belongs to the framework.

    def get_output(self, state: AgentState) -> dict[str, Any]:
        """Extend the base envelope with the structured `citations` field.

        The per-claim source citation list IS this template's product
        differentiator (a grounded answer with a per-claim source citation),
        not a side note, so it is surfaced as a real (decoded) Python value on
        top of the base `output`/`status`/`trace_id`/`correlation_id`/
        `node_history` envelope from AgentBaseGraph.get_output().

        Fail-closed, twice over:
          1. Only when the terminal state.status is SUCCESS is the
             `citations` field attached at all - a non-SUCCESS run (trust-gate
             denial, blocked output, subgraph error, ...) returns only the
             base envelope.
          2. Even on SUCCESS, `citations` is re-scanned through the same
             `_security_gate_output()` used by PostProcessNode before it is
             surfaced - PostProcessNode only gated `result`; this second pass
             guards the field get_output() adds on top of that. A violation
             here flips status to ERROR and withholds `citations`, never a
             partial/best-effort payload.
        """
        base: dict[str, Any] = cast(dict[str, Any], super().get_output(state))

        if state.get("status") != AgentStatus.SUCCESS.value:
            return base

        # A run that completed without processing the request has no structured
        # product to surface. SUCCESS reports that the run reached a defined end
        # safely, not that an answer was produced - so the structured fields are
        # withheld here exactly as they are on a non-SUCCESS status. Only the
        # sentence saying what to correct is returned.
        if state.get("error_code"):
            return base

        citations = from_json(state.get("citations"), []) or []

        violation = _security_gate_output(citations)
        if violation:
            base["status"] = AgentStatus.ERROR.value
            return base

        base["citations"] = citations
        return base


# Back-compat alias - config/agent.yaml declares the dotted class path, and
# src/api/server.py imports the class directly. Keep both names pointing at
# the agent.
Graph = AcademicCurriculumKnowledgeAgent
