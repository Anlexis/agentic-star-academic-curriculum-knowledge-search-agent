# Test Specification — EDU-C2-005

**Template ID:** EDU-C2-005
**Template Name:** AcademicCurriculumKnowledgeAgent
**Category:** Cat 2 (nested RAG)

This document is the contract the shipped test code implements
(`tests/unit/` + `tests/proof_of_boundary/`).

## 1. Scope & Invocation Conventions

- Per-node unit tests for the 5 inner domain nodes + the 2 outer gate nodes.
- Manifest/config consistency (`config/agent.yaml` + `config/config.yaml`
  <-> code) and seeded-KB integrity.
- Retrieval quality (golden queries over `config/kb/edu_regulatory_kb.json`).
- Inner-graph (`DomainWorkflowGraph`) and outer-graph
  (`AcademicCurriculumKnowledgeAgent`) composition / integration.
- Proof-of-Boundary (PoB): import isolation, State msgpack safety, invoke
  order (PB-6), HITL propagation (PB-7, conditional), server boot, and
  end-to-end business behaviour through the real ASGI `/invoke`.

**Trust-gate invocation canon.** Every per-node test invokes the node via
`node(state)` — through `BaseNode.__call__`, which runs the trust gate ->
PII input mask -> `execute()` -> output gate — never a bare
`node.execute(state)`. The state builder sets `caller_trust_level` to
`TrustLevel.VERIFIED_EXTERNAL.value` for the two outer gate slots
(PreProcessNode / PostProcessNode — the manifest's declared caller level)
and `TrustLevel.ANONYMOUS.value` for the five inner domain nodes.

**Node contract.** Every node in this template implements exactly
`execute(self, state) -> dict` — no `config` parameter. RetrieveNode /
RerankFilterNode config overrides (`kb_path`, `top_k`, `score_threshold`) are
exercised exclusively through the state-seeded `retrieval_config` field, via
`node(state)`, matching how `DomainWorkflowGraph._extra_initial_state()` feeds
them at runtime.

**Input-mask expectations.** The framework input gate masks
`user_input`/`validated_input`/`llm_response` (e-mail, phone/SSN/CC digit
groups, Title-Case name bigrams — each word needs >=1 lowercase letter, so an
all-caps acronym like `MEXT` does not match) to `[MASKED]` before `execute()`
runs; confirmed empirically against the installed SDK's
`shared.security.pii_detector.detect_pii()`. Positive-path payloads are
therefore lowercase, PII-free regulatory phrasing; intentional-PII tests
assert the raw identifier is gone and the masked marker (`[MASKED]`, or
`[REDACTED]` for the node's own long-numeric-ID screen) is present.
Domain fields (`grounded_answer`, `formatted_answer`, `retrieved_documents`,
…) are not input-mask scan targets.

**Audit muting.** `shared.*` is never sys.modules-stubbed (the framework
imports `shared.security` at load time). The domain audit emitter is muted
via an autouse fixture patching `src.nodes.<mod>.emit_trace_event`; the audit
assertion test re-patches the same attribute with a spy and asserts on
`call.args[1]` (the event payload).

## 2. Unit Test Cases

### 2.1 PreProcessNode (outer pre_process slot) — `test_pre_process_node.py`

| ID | Case | Input | Expected |
|----|------|-------|----------|
| PRE-01 | Valid query | lowercase EDU regulatory question | `status=SUCCESS`, `validated_input` set, `enriched_context` carries channel/source |
| PRE-02 | Empty input | `""` / whitespace | `status=SUCCESS` + `error_code=EMPTY_INPUT`, `error_log` non-empty, no `validated_input` — the caller can send a question and retry |
| PRE-03 | Missing / non-string input | `user_input` absent; dict payload | `status=SUCCESS` + `error_code=EMPTY_INPUT` |
| PRE-04 | Identifier screen | long ungrouped ID (11 digits) -> node screen; e-mail / grouped 4-4-4 digits / Title-Case bigram -> framework mask | raw identifier absent from `validated_input`; `[REDACTED]` (node screen) / `[MASKED]` (framework mask) present |
| PRE-05 | Channel lock | free-text / non-string `channel` in `input_context` | `enriched_context.channel` locked to `"unknown"` (inert-identifier rule; rejected value never echoed) |
| PRE-08 | Audit | valid query | `pre_process_complete` emitted; payload (`call.args[1]`) carries `input_chars` |

### 2.2 InputValidateNode (inner node 1; caller-data contract gate) — `test_input_validate_node.py`

| ID | Case | Input | Expected |
|----|------|-------|----------|
| VAL-01 | Plain text | free-text query | whole string becomes `search_query`; filters `{category: None, top_k: None}` |
| VAL-02 | Whitespace | ragged spacing/newlines | collapsed to single spaces |
| VAL-03 | JSON envelope | `{"query","category","top_k"}` | all three parsed; `question` alias accepted; category lower-cased/stripped |
| VAL-04 | Malformed JSON | `{`-prefixed non-JSON | treated as plain-text query + parse note |
| VAL-05 | Invalid top_k (both channels) | out-of-range int (0/21/99/-5), bool, float, numeric string, `"many"`, NaN/Infinity (string or raw JSON), over-magnitude | `status=SUCCESS` + `error_code=INVALID_REQUEST`, field-naming message (`top_k` named, value NEVER echoed), no `search_query`/`query_filters` produced — retrieval path fails CLOSED |
| VAL-06 | Valid boundary top_k | 1 / 20 | accepted verbatim |
| VAL-05b | A correctable rejection does not end the conversation | bad `top_k`, empty, whitespace, bad `category`, out-of-range `top_k` (via `invoke()`) | `status=success`; the response names what to change; no answer body; refused instruction-override content still terminates with ERROR |
| VAL-07 | Invalid category (both channels) | unknown slug, spaces/hyphen/markup/over-length, non-string | `status=SUCCESS` + `error_code=INVALID_REQUEST`, field-naming message (value NEVER echoed) — retrieval path fails CLOSED |
| VAL-08 | Oversize query | > 2000 chars | truncated to 2000 + note |
| VAL-09 | Empty request | `""` | `search_query=""` + "empty request" note (non-fatal) |
| VAL-10 | input_context channel | `{"category","top_k"}` via state `input_context` | validated with the same rules; wins over the envelope on overlap; an invalid value on either channel rejects (no silent fallback); non-mapping `input_context` rejects |
| — | JSON-string state | any | `query_filters` is a JSON string, never a bare dict |

### 2.3 RetrieveNode (inner node 2) — `test_retrieve_node.py`

| ID | Case | Input | Expected |
|----|------|-------|----------|
| RET-01 | Happy path | MEXT credit-hour query | top-1 candidate is `kb-001` |
| RET-02 | Ordering | MEXT credit-hour query | scores sorted desc (non-strict); all > 0 |
| RET-03 | Entry shape | any hit | keys `{id,title,category,source,score,excerpt}`; excerpt <= 400 chars |
| RET-04 | Category filter | `query_filters.category="niad_qe_accreditation"` | only that-category entries; top-1 `kb-003` |
| RET-05 | Empty query | `""` | no candidates |
| RET-07 | State `retrieval_config` via `node(state)` | bogus `retrieval_config.kb_path` | `[]` + "not readable" note |
| RET-08 | Notes accumulation | prior `intake_notes` | appended, never clobbered |

### 2.4 RerankFilterNode (inner node 3) — `test_rerank_filter_node.py`

| ID | Case | Input | Expected |
|----|------|-------|----------|
| RRF-01 | Relevance floor | scores 0.9 / 0.1 | 0.1 dropped (default 0.25 floor) |
| RRF-02 | State `retrieval_config` threshold override | `score_threshold=0.5` | 0.3 dropped |
| RRF-03 | State `retrieval_config` top_k override | `top_k=1` | one survivor, highest score |
| RRF-04 | Category boost | matching category | +0.1, re-ranked ahead |
| RRF-05 | Boost cap | 0.95 + boost | capped at 1.0 |
| RRF-06 | Caller top_k | stricter (1) wins; looser (10) does not widen | enforced |
| RRF-07 | Garbage entries | non-dict / uncoercible score | skipped / coerced to 0.0 and dropped |
| RRF-08 | Tie-break | equal scores | deterministic id-ascending order |

### 2.5 GenerateAnswerNode (inner node 4) — `test_generate_answer_node.py`

| ID | Case | Input | Expected |
|----|------|-------|----------|
| GEN-01 | Citation markers | 2 ranked passages | `[1]`/`[2]` markers with titles |
| GEN-02 | Static lead | any query | lead sentence is static; the caller's query is NEVER rendered into the answer (output-injection guard) |
| GEN-03 | Citations list | ranked passages | refs 1..n mirror ranked order; id/title/source carried |
| GEN-04 | Groundedness | single passage | answer body traces to ranked passages only |
| GEN-05 | No coverage | empty/missing `ranked_documents` | escalation answer; `citations=[]` |

### 2.6 OutputFormatNode (inner node 5, terminal) — `test_output_format_node.py`

| ID | Case | Input | Expected |
|----|------|-------|----------|
| FMT-01 | Full compose | body + citations | header + body + `## Sources` rows + advisory disclaimer; `status=SUCCESS` |
| FMT-02 | Blank source | citation without source | no `()` suffix |
| FMT-03 | Disclaimer | every input | disclaimer rides with every answer |
| FMT-04 | No citations | empty list | explicit "- none (…)" sources line |
| FMT-05 | Missing body | no `grounded_answer` | fallback text; `status=SUCCESS` |

### 2.7 PostProcessNode (outer post_process slot; output gate) — `test_post_process_node.py`

| ID | Case | Input (`result`) | Expected |
|----|------|------------------|----------|
| POST-01 | Clean output | normal KB answer | passes through (disclaimer-enforced), `status=SUCCESS` |
| POST-02 | Empty result | `""` | forwarded as-is, `status=SUCCESS` (non-fatal) |
| POST-03..06 | Credential leak | `sk-` API key / `password=` assignment / JWT (built at runtime) / Bearer token | `formatted_output` + `result` replaced with the sanitised stub, `status=ERROR`, raw secret absent from both |
| POST-07 | Nested-structure scan | clean dict/list payload; violation buried dict->list->dict | clean payload passes through unblocked; the 3-level-deep violation is still caught (recursive scan) |
| POST-08 | Disclaimer enforcement | clean answer WITHOUT the disclaimer | disclaimer appended + `output_disclaimer_enforced` audit event; already-composed answers unchanged; the blocked stub is never disclaimer-stamped |

### 2.8 Manifest / runtime-config consistency — `test_config_manifest.py`

| ID | Case | Expected |
|----|------|----------|
| CFG-01 | Identity | manifest root `id` = `EDU-C2-005`, `namespace` = `edu`, `enabled` = true (flat schema — no nested `agent:` block) |
| CFG-02 | Class contract | manifest `class` is the dotted path to the graph.py agent class; `name` matches |
| CFG-03 | Classification | Cat 2 / EDU / RAGAgent; `generation_mode=deterministic`; `requires.secrets=[]`, `requires.extras=[]` (no LLM client constructed) |
| CFG-04 | Trust level | manifest `VERIFIED_EXTERNAL` == PreProcessNode & PostProcessNode `required_trust_level` |
| CFG-05 | Runtime params | config.yaml `max_retry` int within the framework ceiling; `timeout_s` positive int; hitl not enabled (PB-7 waiver contract) |
| CFG-06 | Retrieval block | config.yaml `top_k`/`score_threshold` mirror node module defaults; `kb_path` exists |
| CFG-07 | `_parent_config()` | forwards config.yaml retrieval + llm blocks; never `{}` |
| CFG-08 | Migration guard | no `retrieval`/`llm`/`config`/`agent` key in the static manifest (dead-config guard) |
| — | KB integrity | JSON list >= 5 entries; unique ids; required keys per entry; every category in `InputValidateNode.VALID_CATEGORIES` |

### 2.9 Retrieval quality (golden queries) — `test_retrieval_quality.py`

| ID | Case | Expected |
|----|------|----------|
| QUAL-01 | 10 golden domain queries (one per seeded KB entry) | expected KB entry is top-1 (kb-001..kb-010) |
| QUAL-02 | Relevance floor | every survivor >= 0.25 |
| QUAL-03 | Citation integrity | every survivor id exists in the seeded KB |
| QUAL-04 | Precision | GPA/academic-standing query keeps ONLY `kb-008` |
| QUAL-05 | Category filter | `curriculum_requirements` filter -> only that-category entries, top-1 `kb-002` |
| QUAL-06 | No coverage | out-of-domain query -> zero survivors |
| QUAL-07 | Escalation answer | no-coverage -> explicit escalation text, no citations |

### 2.10 Trust matrix — `test_trust_gate.py`

| Case | Expected |
|------|----------|
| ANONYMOUS on inner node | admitted (all 5 inner nodes, runtime-proven) |
| ANONYMOUS on pre_process / post_process | denied: `status=ERROR`, "trust gate denied" in `error_log`, execute-only keys ABSENT |
| VERIFIED_EXTERNAL on outer slots | admitted |
| Output-gate behaviour | credential-like output blocked (behavioural assertion: ERROR + token gone — never the gate's wording); nested-structure violation caught |
| Static matrix | outer slots VERIFIED_EXTERNAL; inner nodes ANONYMOUS |

### 2.11 Marketplace entry point — `test_cli_entry_point.py`

| ID | Case | Expected |
|----|------|----------|
| CLI-01 | `cli.py` imports | module loads; `run_agent_marketplace`, `load_agent_config` and `AcademicCurriculumKnowledgeAgent` are present |
| CLI-02 | override seam ships empty | `extend_config == {}`; a stray value would silently outrank `config/config.yaml` on the Marketplace path only |
| CLI-03 | the runner receives what the image's CMD would send | executing `cli.py` as `__main__` with the runner replaced captures the call: the graph class, `agent_name`, `namespace`, and every value declared in `config/config.yaml`. Loading the module alone never runs that block, so a wrong class or a dropped config there would otherwise ship unnoticed |

`cli.py` is imported by no other module, so nothing else in the suite would
notice if its import path, graph class or config assembly broke; the image
would build and fail only when the Pod starts. Skipped where the platform
events package is absent.

### 2.12 Framework compliance — `test_framework_compliance_tc06_tc07.py`

| ID | Case | Expected |
|----|------|----------|
| TC-06 | Overriding the default input gate | raises TypeError at class definition |
| TC-07 | Overriding the default output gate | raises TypeError at class definition |

## 3. Integration / Composition

### 3.1 Inner graph — `test_domain_workflow_graph.py`

| ID | Case | Expected |
|----|------|----------|
| INT-01 | Composition | inherits `BaseGraph`; registers exactly the 5 domain nodes; no initialize/finalize |
| INT-02 | Config + context seeding | `_extra_initial_state()` republishes the retrieval block as the JSON-string `retrieval_config` AND seeds the bridged `input_context` |
| INT-03 | Output shape | `get_output()` emits `formatted_answer`/`citations`/`status`/`error_log`/… (the merge contract); `route()` -> END on error |
| INT-04 | Inner e2e | full inner `invoke()` -> SUCCESS; formatted answer + disclaimer + kb-001 citation; inner `node_history` = the 5 domain nodes in linear order |

### 3.2 Outer graph + e2e — `test_graph_composition.py`

| ID | Case | Expected |
|----|------|----------|
| INT-05 | Outer composition | inherits `AgentBaseGraph` directly; `Graph` alias; `add_edges()` NOT overridden |
| INT-06 | Backbone slots | compile() fills all 5; pre/main/post are PreProcessNode / CurriculumKnowledgeGraphNode / PostProcessNode |
| INT-07 | `get_subgraph()` | returns `DomainWorkflowGraph` carrying the forwarded retrieval config |
| INT-08 | `extract_input()` | prefers `validated_input`, falls back to `user_input` |
| INT-09 | `merge_output()` | inner `formatted_answer` -> outer `knowledge_base_answer` AND `result`; `citations`/`status` mapped; changed keys only |
| INT-10 | Config fallback | `_parent_config()` never `{}` even with an unreadable config/config.yaml |
| INT-11 | e2e happy path | VERIFIED_EXTERNAL invoke -> SUCCESS; `output` = gated formatted answer; PostProcessNode traversed |
| INT-12 | e2e trust denial | ANONYMOUS invoke -> ERROR; empty `output`; PostProcessNode NOT traversed |
| INT-13 | Context bridge | `extract_input()` stashes `input_context` for the inner graph's `_extra_initial_state()` |
| — | JSON-string helpers | `to_json`/`from_json` round-trip; None/malformed handling |

## 4. Proof-of-Boundary

| ID | Case | Expected |
|----|------|----------|
| PB-IMPORT | `test_import_isolation.py` | no platform-internal SDK import anywhere under `src/` |
| PB-STATE | `test_state_safety.py` | `State` has no credential-named fields and no `BaseModel` / `InvocationContext` annotations |
| PB-6 | `test_pb_invoke_order.py` | full `Graph().invoke()` with `InvocationContext(caller_trust_level=VERIFIED_EXTERNAL)` (never `for_internal()`) over the payload byte-equal to `deploy/invoke_payload.json`'s `input` -> SUCCESS with outer `node_history` exactly `[InitializeNode, PreProcessNode, CurriculumKnowledgeGraphNode, PostProcessNode, FinalizeNode]`. Also asserts the negative S-1 path (TC-08): an under-privileged caller is refused before `execute()` runs and before the normal lifecycle events are emitted |
| PB-5 | `test_state_safety.py` | **Auto-waived — checkpointing disabled** (`config/config.yaml` enables neither `memory_enabled` nor `hitl.enabled`); the conditional gate and the non-lossy traversal helper ship with the stub |
| PB-7 | `test_pb7_hitl_interrupt_propagation.py` | **Auto-waived — non-HITL** (no `hitl.enabled: true` declared); conditional skip-stub retained |
| PB-BOOT | `test_server_boot.py` | `import src.api.server` does not raise; module-level agent is this template's class, compiled; fresh ctor->`compile()` fills the 5 backbone slots; `/health` reports the agent |
| PB-E2E | `test_invoke_e2e.py` | through the REAL ASGI `/invoke` (Bearer auth): grounded cited answer from a caller question (never a stub baseline); `input_context` `top_k`/`category` reach the inner graph and change the outcome (context-bridge proof); no-coverage degrade path; fail-closed rejections for malformed caller parameters incl. the per-field non-finite matrix (`NaN`/`Infinity`/bool/float/out-of-range, string and raw-JSON forms) with the rejected value never echoed; prompt-injection payload refused with nothing published (behavioural assertion); every success output carries the documented schema (Sources + disclaimer, no credential-shaped token); wrong Bearer token -> 401; oversized `input_context` -> 413 |

> **Gate checklist:** PB-IMPORT, PB-STATE, PB-6, PB-BOOT and PB-E2E are
> mandatory. PB-5 applies only when checkpointing is enabled and the installed
> framework exposes its ingress hooks — neither holds here, so it is
> **Auto-waived — checkpointing disabled**. PB-7 applies only to HITL-enabled templates — this template is
> non-HITL, so PB-7 is **Auto-waived — non-HITL** and its skip must not block
> the gate.

## 5. Test Execution Summary

> **Pending re-run.** The figures below predate the Marketplace entry point work.
> `test_cli_entry_point.py` (§2.11) and the PB-5 / PB-6 additions were added after
> this run and have not been executed locally — the framework wheel is not installed
> in the authoring environment. **They are not yet verified anywhere**; this summary
> is updated once a pipeline run has executed them.


- Execution date: 2026-08-25
- Runner: real SDK (`agenticstar-agentcore==1.0.1`), `python -m pytest tests/`
- Total tests (full `tests/` tree): 214
- Pass: 213 / Fail: 0 / Skip: 1 (PB-7 conditional stub — auto-waived, non-HITL)
- `tests/proof_of_boundary/` subset: 32 total — Pass: 31 / Fail: 0 / Skip: 1
- Determinism: no LLM, no network; retrieval + answer assembly are rule-based
