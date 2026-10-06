"""AgentCore Platform v1.0"""

# State must be a flat TypedDict - never Pydantic BaseModel. LangGraph
# checkpoints use msgpack serialization; Pydantic objects cause silent
# corruption.  Extend AgentState with agent-specific fields only.  Do NOT
# add credentials, secrets, or Pydantic models.
#
# Msgpack safety: structured fields (dict / list[dict]) are stored as JSON
# STRINGS, not bare Python containers - a bare dict/list in a checkpointed
# State field corrupts silently. Producers serialize with to_json() on
# write; consumers deserialize with from_json() on read.
#
# EDU-C2-005 - Academic Curriculum Knowledge Agent (Cat 2 RAG, nested).
# Two-layer nested Cat 2 graph: outer backbone (AgentBaseGraph) + inner
# domain workflow (BaseGraph).  Fields below cover both layers.
#
# PII / confidentiality note: direct identifiers (long numeric IDs, e-mail)
# in the query payload are surface-stripped by PreProcessNode before any
# field is written to State.  Only the normalised search query,
# KB passage summaries, and the final grounded answer are persisted - never
# raw personal identifiers. This template's KB is regulatory/administrative
# content only (MEXT / NIAD-QE / institutional policy text) - no student
# records are read, stored, or referenced anywhere in State (student-record
# processing is out of scope for this template indefinitely).

import json
from typing import Any, Optional

from framework.schemas.agent_state import AgentState


def to_json(value: Any) -> Optional[str]:
    """Serialize a dict/list State field to a JSON string (msgpack safety).

    None passes through unchanged so an 'unset' field stays distinguishable
    from an empty container.
    """
    if value is None:
        return None
    return json.dumps(value, ensure_ascii=False)


def from_json(value: Optional[str], default: Any = None) -> Any:
    """Deserialize a JSON-string State field back to its dict/list.

    None / empty / malformed input -> the supplied ``default`` so a missing or
    corrupt field is non-fatal for the consuming node.
    """
    if not value:
        return default
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return default


class State(AgentState):
    """Flat TypedDict for EDU-C2-005.

    All shared fields (user_input, status, session_id, node_history,
    error_log, hitl_*, etc.) are inherited from AgentState.
    Domain fields are Optional and unset at graph initialisation - nodes
    write them as the run progresses (runtime TypedDicts do not enforce key
    presence; consumers always read via state.get()).
    """

    # ------------------------------------------------------------------
    # Outer layer - set by PreProcessNode / CurriculumKnowledgeGraphNode.merge_output
    # ------------------------------------------------------------------

    # Identifier-stripped, validated query payload produced by PreProcessNode.
    # Raw input is NOT persisted beyond PreProcessNode.
    validated_input: Optional[str]

    # Final knowledge-base search answer, mapped from the inner graph's
    # formatted_answer output via merge_output.
    knowledge_base_answer: Optional[str]

    # ------------------------------------------------------------------
    # Inner layer - domain nodes (DomainWorkflowGraph)
    # ------------------------------------------------------------------

    # InputValidateNode outputs
    # Normalised free-text search query (whitespace-collapsed, length-capped).
    # Never rendered into the answer body - it drives retrieval only.
    search_query: Optional[str]

    # JSON STRING (to_json) of parsed structured query params. Deserialised
    # dict shape: {"category": str | None, "top_k": int | None}. category is
    # one of "mext_course_of_study" / "niad_qe_accreditation" /
    # "curriculum_requirements" / "institutional_policy".
    # Consumers (RetrieveNode, RerankFilterNode) read it back via from_json().
    query_filters: Optional[str]

    # Runtime `retrieval` block (config/config.yaml) forwarded by
    # CurriculumKnowledgeGraphNode._parent_config() ->
    # DomainWorkflowGraph._extra_initial_state().
    # JSON STRING (to_json) of {"top_k": int, "score_threshold": float,
    # "kb_path": str}. Consumers (RetrieveNode, RerankFilterNode) read it
    # back via from_json(). This is the ONLY route config reaches these
    # nodes - execute() takes no `config` parameter.
    retrieval_config: Optional[str]

    # RetrieveNode output
    # JSON STRING (to_json) of scored KB candidates. Deserialised shape:
    # list[dict], each entry {"id": str, "title": str, "category": str,
    # "source": str, "score": float, "excerpt": str}.
    # Consumers (RerankFilterNode) read it back via from_json().
    retrieved_documents: Optional[str]

    # RerankFilterNode output
    # JSON STRING (to_json) of reranked + threshold-filtered passages, capped
    # at top_k. Same entry shape as retrieved_documents.
    # Consumers (GenerateAnswerNode) read it back via from_json().
    ranked_documents: Optional[str]

    # GenerateAnswerNode outputs
    # Rule-assembled grounded answer body with numbered citation markers.
    grounded_answer: Optional[str]

    # JSON STRING (to_json) of citations. Deserialised shape: list[dict],
    # each entry {"ref": int, "id": str, "title": str, "source": str}.
    # Consumers (OutputFormatNode) read it back via from_json(). Also
    # re-surfaced at the outer layer by merge_output() so
    # AcademicCurriculumKnowledgeAgent.get_output() can read it without
    # reaching into the inner graph.
    citations: Optional[str]

    # OutputFormatNode output
    # Final formatted answer (body + sources + advisory disclaimer). Written by
    # OutputFormatNode; surfaced to the outer graph via get_output() ->
    # merge_output().
    formatted_answer: Optional[str]

    # Validation / parse notes accumulated during intake (no PII).
    # JSON STRING (to_json) of list[str].
    intake_notes: Optional[str]

    # ------------------------------------------------------------------
    # Degraded completion marker
    # ------------------------------------------------------------------

    # Set when the run completes WITHOUT producing an answer because the
    # caller's request could not be accepted as written - a rejection the
    # caller can correct and retry (an unusable structured parameter, an
    # empty query). The run still completes: no retrieval is performed, no
    # answer is assembled, and the domain audit event for the rejection is
    # still emitted. Carrying this as a completion marker rather than a
    # terminal error is what lets the caller see the reason and send a
    # corrected request on the same conversation.
    #
    # A breach of a contract the caller cannot influence (a field an earlier
    # step is required to set) is NOT reported here - that stays a terminal
    # error so it is not mistaken for something the caller can fix.
    #
    # Once set, every later domain node passes through without doing work,
    # and the value is carried across the inner/outer boundary by
    # get_output() and merge_output().
    error_code: Optional[str]

    # ------------------------------------------------------------------
    # Tracing / audit - framework-managed; do NOT write from node code
    # ------------------------------------------------------------------

    trace_id: Optional[str]
    correlation_id: Optional[str]
    # node_history inherited from AgentState; listed here for clarity
    # node_history: Optional[List[str]]
