"""AgentCore Platform v1.0"""

# EDU-C2-005 - RetrieveNode
# Domain node 2: deterministic keyword retrieval over the seeded EDU
# regulatory knowledge base (config/kb/edu_regulatory_kb.json: MEXT
# course-of-study guidelines, NIAD-QE accreditation standards, curriculum
# requirements, institutional policies). v1 is fully deterministic - no
# embedding model or vector store; the retrieval contract
# (retrieved_documents JSON) is store-agnostic so a later vector-store
# upgrade only swaps this node's internals.
#
# Config: reads `top_k` (and `kb_path`) from the state-seeded
# `retrieval_config` field (republished by
# DomainWorkflowGraph._extra_initial_state(), forwarded from
# CurriculumKnowledgeGraphNode._parent_config()), falling back to module
# defaults that mirror the `retrieval` block in config/config.yaml.
# execute() takes no `config` parameter.
#
# Wired by the inner graph (DomainWorkflowGraph).
# Returns only changed state keys (partial dict).

import json
import re
from pathlib import Path
from typing import Any, ClassVar, Dict, List

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_state import AgentState
from framework.schemas.trust_level import TrustLevel
from shared.utils.audit_logger import emit_trace_event
from src.services.progress import emit_progress
from src.schemas.state import from_json, to_json
from framework.schemas.agent_status import AgentStatus

# Defaults mirror the `retrieval` block in config/config.yaml.
_DEFAULT_RETRIEVAL: Dict[str, Any] = {
    "top_k": 4,
    "score_threshold": 0.25,
    "kb_path": "config/kb/edu_regulatory_kb.json",
}

# Repo root: src/nodes/retrieve_node.py -> parents[2].
_REPO_ROOT = Path(__file__).resolve().parents[2]

# Minimal stopword set for query tokenisation (deterministic, no NLP deps).
_STOPWORDS = frozenset(
    {
        "the",
        "a",
        "an",
        "and",
        "or",
        "of",
        "to",
        "in",
        "on",
        "for",
        "is",
        "are",
        "be",
        "with",
        "under",
        "what",
        "which",
        "when",
        "how",
        "do",
        "does",
        "must",
        "should",
        "before",
        "after",
        "by",
        "at",
        "from",
        "that",
        "this",
        "it",
        "as",
        "was",
        "were",
        "can",
        "may",
        "any",
    }
)

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Per-field match weights: a query token found in the title counts more than
# one found only in the body content.
_TITLE_WEIGHT = 1.0
_TAG_WEIGHT = 0.8
_CONTENT_WEIGHT = 0.5

# Excerpt length carried into retrieved_documents (keeps State small).
_EXCERPT_CHARS = 400


def _tokenize(text: str) -> List[str]:
    """Lowercase alphanumeric tokens, stopwords and 1-2 char noise removed."""
    return [t for t in _TOKEN_RE.findall(text.lower()) if len(t) > 2 and t not in _STOPWORDS]


def _resolve_retrieval_config(state: AgentState) -> Dict[str, Any]:
    """Effective retrieval config: state-seeded retrieval_config > defaults.

    No `config` parameter is read here - execute() takes only `state`.
    Config knobs reach this node exclusively via the state-seeded
    `retrieval_config` field.
    """
    effective = dict(_DEFAULT_RETRIEVAL)  # local copy - never mutate the module default
    from_state = from_json(state.get("retrieval_config"), None)
    if isinstance(from_state, dict):
        effective.update(from_state)
    return effective


def _load_kb(kb_path: str) -> tuple[List[Dict[str, Any]], List[str]]:
    """Load the seeded KB JSON. Missing / malformed file degrades gracefully."""
    notes: List[str] = []
    path = Path(kb_path)
    if not path.is_absolute():
        path = _REPO_ROOT / path
    try:
        entries = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError):
        notes.append(f"RetrieveNode: knowledge base not readable at {kb_path}.")
        return [], notes
    if not isinstance(entries, list):
        notes.append("RetrieveNode: knowledge base root must be a JSON list.")
        return [], notes
    return [e for e in entries if isinstance(e, dict)], notes


def _score_entry(entry: Dict[str, Any], query_tokens: List[str]) -> float:
    """Per-entry relevance: best field-weight per query token, averaged."""
    if not query_tokens:
        return 0.0
    title_tokens = set(_tokenize(str(entry.get("title", ""))))
    tag_tokens = set(_tokenize(" ".join(str(t) for t in entry.get("tags", []))))
    content_tokens = set(_tokenize(str(entry.get("content", ""))))
    total = 0.0
    for token in query_tokens:
        if token in title_tokens:
            total += _TITLE_WEIGHT
        elif token in tag_tokens:
            total += _TAG_WEIGHT
        elif token in content_tokens:
            total += _CONTENT_WEIGHT
    return round(total / len(query_tokens), 4)


class RetrieveNode(FunctionNode):
    """Score the seeded KB against the search query and emit candidates.

    Input state keys:
        search_query:     normalised query (from InputValidateNode)
        query_filters:    JSON dict with optional category filter
        retrieval_config: forwarded runtime retrieval block (JSON)

    Output state keys (partial dict):
        retrieved_documents: JSON list of scored candidates (score desc)
        intake_notes:        (on KB anomalies) JSON list[str]
    """

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.ANONYMOUS

    def execute(self, state: AgentState) -> dict[str, Any]:
        # The request was already found unacceptable upstream: this run
        # completes without an answer, so there is nothing for this step to
        # do. Returning the marker keeps it on the node's own result dict,
        # which is what the output gate inspects.
        marker = state.get("error_code")
        if marker:
            return {"status": AgentStatus.SUCCESS.value, "error_code": marker}

        emit_progress("Searching the knowledge base...")
        query = state.get("search_query") or state.get("validated_input") or state.get("user_input", "")
        filters = from_json(state.get("query_filters"), {}) or {}
        retrieval_cfg = _resolve_retrieval_config(state)

        try:
            top_k = int(retrieval_cfg.get("top_k", _DEFAULT_RETRIEVAL["top_k"]))
        except (TypeError, ValueError):
            top_k = int(_DEFAULT_RETRIEVAL["top_k"])
        top_k = max(1, min(20, top_k))

        entries, notes = _load_kb(str(retrieval_cfg.get("kb_path", _DEFAULT_RETRIEVAL["kb_path"])))

        category = filters.get("category")
        if category:
            entries = [e for e in entries if str(e.get("category", "")).lower() == str(category).lower()]

        query_tokens = _tokenize(query if isinstance(query, str) else "")

        candidates: List[Dict[str, Any]] = []
        for entry in entries:
            score = _score_entry(entry, query_tokens)
            if score <= 0.0:
                continue
            candidates.append(
                {
                    "id": str(entry.get("id", "")),
                    "title": str(entry.get("title", "")),
                    "category": str(entry.get("category", "")),
                    "source": str(entry.get("source", "")),
                    "score": score,
                    "excerpt": str(entry.get("content", ""))[:_EXCERPT_CHARS],
                }
            )

        # Deterministic ordering: score desc, then id asc for stable ties.
        candidates.sort(key=lambda c: (-c["score"], c["id"]))
        # Keep a candidate pool wider than top_k - RerankFilterNode makes
        # the final cut after the category boost + threshold.
        pool_size = max(top_k * 3, 10)
        candidates = candidates[:pool_size]

        # Domain audit: retrieval pass completed.
        emit_trace_event(
            "retrieve_complete",
            {
                "candidates": len(candidates),
                "kb_entries": len(entries),
                "query_tokens": len(query_tokens),
                "top_k": top_k,
            },
            state,
        )

        out: Dict[str, Any] = {"retrieved_documents": to_json(candidates)}
        if notes:
            # Append to (never clobber) the notes accumulated upstream.
            prior = from_json(state.get("intake_notes"), []) or []
            out["intake_notes"] = to_json(list(prior) + notes)
        return out
