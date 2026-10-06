# EDU-C2-005 — Unit Tests: RetrieveNode (inner domain node 2)
#
# Invocation canon: node(state) via BaseNode.__call__ with an ANONYMOUS caller.
# Node contract: execute(self, state) -> dict is
# the ONLY signature — there is no `config` parameter to pass, so every test
# below reaches config exclusively through the state-seeded `retrieval_config`
# field (the same route DomainWorkflowGraph._extra_initial_state() uses at
# runtime), never a direct execute(state, config=...) call.
#
# All scores below were confirmed empirically against the real node code and
# the real seeded config/kb/edu_regulatory_kb.json (not hand-computed).
#
# Mirrors docs/03_test_spec.md S2.3 (RET-01..RET-08).
# Deterministic — keyword scoring over the seeded KB; no LLM, no network.
# framework.* / src.* imports only.

from framework.schemas.trust_level import TrustLevel

from src.nodes.retrieve_node import RetrieveNode
from src.schemas.state import from_json, to_json

_MEXT_QUERY = "What does the MEXT guideline say about credit-hour requirements for " "degree programs?"


def _make_state(query=_MEXT_QUERY, **extra) -> dict:
    state = {
        "search_query": query,
        "caller_trust_level": TrustLevel.ANONYMOUS.value,
        "node_history": [],
        "error_log": [],
        "session_id": "unit-session",
        "execution_time": {},
    }
    state.update(extra)
    return state


class TestRetrieveHappyPath:
    def test_ret_01_top_hit_is_credit_hour_entry(self):
        result = RetrieveNode()(_make_state())
        docs = from_json(result["retrieved_documents"])
        assert docs, "expected candidates for the MEXT credit-hour query"
        assert docs[0]["id"] == "kb-001"

    def test_ret_02_scores_sorted_descending(self):
        docs = from_json(RetrieveNode()(_make_state())["retrieved_documents"])
        scores = [d["score"] for d in docs]
        assert scores == sorted(scores, reverse=True)
        assert all(s > 0.0 for s in scores)

    def test_ret_03_entry_shape_and_excerpt_cap(self):
        docs = from_json(RetrieveNode()(_make_state())["retrieved_documents"])
        for doc in docs:
            assert set(doc.keys()) == {"id", "title", "category", "source", "score", "excerpt"}
            assert len(doc["excerpt"]) <= 400

    def test_retrieved_documents_is_json_string(self):
        # List-shaped State fields travel as JSON strings.
        result = RetrieveNode()(_make_state())
        assert isinstance(result["retrieved_documents"], str)


class TestRetrieveFilters:
    def test_ret_04_category_filter_restricts_pool(self):
        state = _make_state(
            query="institutional accreditation quality review standards",
            query_filters=to_json({"category": "niad_qe_accreditation", "top_k": None}),
        )
        docs = from_json(RetrieveNode()(state)["retrieved_documents"])
        assert docs, "niad_qe_accreditation category has seeded entries"
        assert {d["category"] for d in docs} == {"niad_qe_accreditation"}
        assert docs[0]["id"] == "kb-003"

    def test_ret_05_empty_query_yields_no_candidates(self):
        docs = from_json(RetrieveNode()(_make_state(query=""))["retrieved_documents"])
        assert docs == []


class TestRetrieveConfigPrecedence:
    """Config plumbing: state-seeded retrieval_config > module defaults.

    C2 retired — there is no passed-config route to test here at all; every
    call goes through node(state), config reaching the node only via the
    state-seeded `retrieval_config` field.
    """

    def test_ret_07_state_retrieval_config_kb_path_override(self):
        state = _make_state(retrieval_config=to_json({"kb_path": "config/kb/bogus.json"}))
        result = RetrieveNode()(state)
        assert from_json(result["retrieved_documents"]) == []
        notes = from_json(result.get("intake_notes"), [])
        assert any("not readable" in n for n in notes)


class TestRetrieveNotesAccumulation:
    def test_ret_08_notes_append_never_clobber(self):
        state = _make_state(
            intake_notes=to_json(["earlier note from input validation"]),
            retrieval_config=to_json({"kb_path": "config/kb/bogus.json"}),
        )
        result = RetrieveNode()(state)
        notes = from_json(result["intake_notes"])
        assert notes[0] == "earlier note from input validation"
        assert len(notes) == 2
