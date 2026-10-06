# EDU-C2-005 — Unit Tests: manifest / runtime-config consistency
#
# Two config files, two jobs:
#   - config/agent.yaml  — the static AgentRegistry manifest: identity, entry
#     point, trust level, compile-time requirements. Every key sits at ROOT
#     level (never nested under an `agent:` block).
#   - config/config.yaml — the runtime parameters: max_retry / timeout_s plus
#     the `retrieval` / `llm` tuning blocks.
#     CurriculumKnowledgeGraphNode._parent_config() forwards the tuning
#     blocks into the inner graph, so these values are LIVE configuration.
#
# These tests pin manifest/config <-> code consistency so a config drift
# fails fast in CI.
#
# Mirrors docs/03_test_spec.md §2.8 (CFG-01..CFG-08).
# Deterministic - no LLM, no network.

import json
import pathlib

import yaml

from framework.schemas.trust_level import TrustLevel

from src.graph.graph import AcademicCurriculumKnowledgeAgent, CurriculumKnowledgeGraphNode
from src.nodes.post_process_node import PostProcessNode
from src.nodes.pre_process_node import PreProcessNode

_ROOT = pathlib.Path(__file__).resolve().parents[2]
_MANIFEST = yaml.safe_load((_ROOT / "config" / "agent.yaml").read_text(encoding="utf-8"))
_RUNTIME = yaml.safe_load((_ROOT / "config" / "config.yaml").read_text(encoding="utf-8"))


class TestManifestIdentity:
    def test_cfg_01_agent_id_is_the_template_id(self):
        # Flat AgentRegistry schema: identity keys sit at manifest ROOT level.
        assert _MANIFEST["id"] == "EDU-C2-005"
        assert _MANIFEST["namespace"] == "edu"
        assert _MANIFEST["enabled"] is True

    def test_cfg_02_declared_class_is_the_graph_class(self):
        # Class-name contract: the manifest's dotted entry point resolves to
        # the src/graph/graph.py agent class the server imports.
        assert _MANIFEST["class"] == "src.graph.graph.AcademicCurriculumKnowledgeAgent"
        assert _MANIFEST["class"].rsplit(".", 1)[-1] == AcademicCurriculumKnowledgeAgent.__name__
        assert _MANIFEST["name"] == AcademicCurriculumKnowledgeAgent().name

    def test_cfg_03_category_and_industry(self):
        assert _MANIFEST["category"] == "Cat 2"
        assert _MANIFEST["industry"] == "EDU"
        assert _MANIFEST["base_type"] == "RAGAgent"
        # No LLM client is constructed anywhere in src/ — the manifest must
        # declare the deterministic generation mode and no extras/secrets.
        assert _MANIFEST["generation_mode"] == "deterministic"
        assert _MANIFEST["requires"] == {"secrets": [], "extras": []}


class TestManifestSecurity:
    def test_cfg_04_required_trust_level_matches_outer_gate_nodes(self):
        declared = TrustLevel(_MANIFEST["required_trust_level"])
        assert declared is TrustLevel.VERIFIED_EXTERNAL
        assert PreProcessNode.required_trust_level is declared
        assert PostProcessNode.required_trust_level is declared

    def test_cfg_05_max_retry_within_framework_ceiling(self):
        max_retry = _RUNTIME["max_retry"]
        assert isinstance(max_retry, int)
        assert 0 <= max_retry < 10  # AgentBaseGraph MAX_RETRY_CEILING

    def test_cfg_05_timeout_is_declared_in_seconds(self):
        timeout_s = _RUNTIME["timeout_s"]
        assert isinstance(timeout_s, int)
        assert timeout_s > 0

    def test_hitl_is_not_enabled(self):
        # PB-7 auto-waiver contract: this template declares no HITL block at all.
        assert (_RUNTIME.get("hitl") or {}).get("enabled", False) is False
        assert "hitl" not in _MANIFEST


class TestRetrievalBlock:
    def test_cfg_06_retrieval_block_matches_node_defaults(self):
        # Node module defaults mirror config/config.yaml — a drift silently
        # changes tuning.
        retrieval = _RUNTIME["retrieval"]
        from src.nodes.retrieve_node import _DEFAULT_RETRIEVAL as retrieve_defaults
        from src.nodes.rerank_filter_node import _DEFAULT_RETRIEVAL as rerank_defaults

        assert retrieval["top_k"] == retrieve_defaults["top_k"] == rerank_defaults["top_k"]
        assert (
            retrieval["score_threshold"] == retrieve_defaults["score_threshold"] == rerank_defaults["score_threshold"]
        )
        assert retrieval["kb_path"] == retrieve_defaults["kb_path"]
        assert (_ROOT / retrieval["kb_path"]).is_file()

    def test_cfg_07_parent_config_forwards_runtime_blocks(self):
        # _parent_config() reads config/config.yaml (NOT the static manifest,
        # which carries no tuning values) and must forward both blocks.
        cfg = CurriculumKnowledgeGraphNode()._parent_config()
        assert cfg["configurable"]["retrieval"] == _RUNTIME["retrieval"]
        assert cfg["configurable"]["llm"] == _RUNTIME["llm"]
        assert cfg["configurable"]["retrieval"], "_parent_config() must never forward an empty retrieval block"

    def test_cfg_08_manifest_carries_no_tuning_blocks(self):
        # Migration guard: a `retrieval`/`llm`/`config` block reappearing in
        # the static manifest would be dead configuration text — the live
        # reader is config/config.yaml.
        for stale_key in ("retrieval", "llm", "config", "agent"):
            assert stale_key not in _MANIFEST, f"tuning key {stale_key!r} belongs in config/config.yaml"


class TestSeededKnowledgeBase:
    def test_kb_is_a_well_formed_entry_list(self):
        entries = json.loads((_ROOT / _RUNTIME["retrieval"]["kb_path"]).read_text(encoding="utf-8"))
        assert isinstance(entries, list)
        assert len(entries) >= 5, "seeded KB must carry a usable corpus"
        for entry in entries:
            assert set(entry.keys()) == {"id", "title", "category", "source", "tags", "content"}
            assert entry["id"] and entry["title"] and entry["content"]

    def test_kb_ids_are_unique(self):
        entries = json.loads((_ROOT / _RUNTIME["retrieval"]["kb_path"]).read_text(encoding="utf-8"))
        ids = [e["id"] for e in entries]
        assert len(ids) == len(set(ids))

    def test_kb_categories_are_within_the_valid_set(self):
        # InputValidateNode.VALID_CATEGORIES is the caller-facing filter contract;
        # every seeded entry's category must be a member (else it is unreachable
        # by a category-filtered request).
        from src.nodes.input_validate_node import VALID_CATEGORIES

        entries = json.loads((_ROOT / _RUNTIME["retrieval"]["kb_path"]).read_text(encoding="utf-8"))
        for entry in entries:
            assert (
                entry["category"] in VALID_CATEGORIES
            ), f"KB entry {entry['id']!r} category {entry['category']!r} not in {VALID_CATEGORIES}"
