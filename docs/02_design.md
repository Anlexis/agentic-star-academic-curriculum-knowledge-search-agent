# Template Design Specification — EDU-C2-005

**Template ID:** EDU-C2-005
**Template Name:** AcademicCurriculumKnowledgeAgent
**Category:** Cat 2 (multi-step domain workflow — RAG pattern)
**Industry:** EDU

## Position in AgentCore Architecture

| Role | Value |
|------|-------|
| Agent Class | `AcademicCurriculumKnowledgeAgent` (alias `Graph`) |
| L1 Base (framework base class) | `AgentBaseGraph` — direct framework inheritance |
| Inner graph base | `BaseGraph` — `DomainWorkflowGraph` |
| Pattern | Cat 2 two-layer nested architecture (outer fixed 5-node backbone + `GraphNode` in the `main` slot wrapping an inner `BaseGraph` domain workflow) |

**Three-Layer Separation:**
- State: flat `TypedDict` composition (no Pydantic — msgpack incompatible);
  structured fields stored as JSON strings via `to_json()` / `from_json()`
- Node: framework inheritance via `FunctionNode` (override
  `execute(self, state) -> dict` only — no `config` parameter)
- Graph: composition (`register_nodes()` for node substitution); outer
  `add_edges()` is NOT overridden

## Purpose

Retrieval-augmented Q&A over a curated EDU regulatory knowledge base: MEXT
course-of-study guidelines, NIAD-QE accreditation standards, curriculum
requirements, and institutional policies. Faculty, curriculum designers, and
administrative/accreditation staff ask a natural-language question and
receive a grounded answer with a per-claim source citation — retrieve →
rerank/filter → grounded answer with citations, behind the mandatory output
security gate. The shipped build is fully deterministic (keyword retrieval +
rule-based grounded answer assembly; no live LLM call — see the
Implementation Note below).

## Architecture Overview

### Outer backbone (AgentBaseGraph)

```
START → initialize → pre_process → main → {route} → post_process → finalize → END
                                     ↓ (retry, max 3)
                                   pre_process
```

| Slot | Class | Responsibility | required_trust_level |
|------|-------|----------------|----------------------|
| initialize | InitializeNode (framework default) | session_id, trust_level, schema_version | — (framework) |
| pre_process | `PreProcessNode` | validate non-empty input; surface-strip direct identifiers (long numeric IDs, e-mail) → `validated_input`; lock the caller `channel` label to an inert identifier | `TrustLevel.VERIFIED_EXTERNAL` |
| main | `CurriculumKnowledgeGraphNode` (`GraphNode`) | delegates to inner `DomainWorkflowGraph`; bridges `input_context`; maps inner `formatted_answer` → outer `result` | — (GraphNode delegation) |
| post_process | `PostProcessNode` | output security gate — module-level `_security_gate_output()` scans `result` (recursively — dict/list/tuple/set) for credentials → ERROR + sanitised stub; independently enforces the standing advisory disclaimer | `TrustLevel.VERIFIED_EXTERNAL` |
| finalize | FinalizeNode (framework default) | response_metadata, total_time_ms | — (framework) |

### Inner graph (DomainWorkflowGraph — BaseGraph, linear)

```
START → input_validate → retrieve → rerank_filter → generate_answer → output_format → END
```

All five inner domain nodes declare `required_trust_level = TrustLevel.ANONYMOUS`
(the external trust gate lives on the outer backbone gate node; a stricter inner
level would deny a real VERIFIED_EXTERNAL invoke at runtime).

| Node | Responsibility | required_trust_level | Input State | Output State |
|------|----------------|----------------------|-------------|---------------|
| `InputValidateNode` | The caller-data contract gate: parse the (possibly JSON-enveloped) query; normalise whitespace; cap length; validate `category` / `top_k` from BOTH channels (in-band envelope + bridged `input_context`) fail-closed — an invalid value stops the retrieval path with a field-naming error, never echoing the value | `TrustLevel.ANONYMOUS` | `validated_input` \| `user_input`, `input_context` | `search_query`, `query_filters`, `intake_notes` (or `error_code` + `error_log`; refused content → ERROR) |
| `RetrieveNode` | Deterministic keyword retrieval over the seeded KB (`config/kb/edu_regulatory_kb.json`): tokenise query, score title/tags/content overlap, apply category filter | `TrustLevel.ANONYMOUS` | `search_query`, `query_filters`, `retrieval_config` | `retrieved_documents`, `intake_notes` |
| `RerankFilterNode` | Rerank candidates (category-match boost), drop entries below `score_threshold`, cap at `top_k` | `TrustLevel.ANONYMOUS` | `retrieved_documents`, `query_filters`, `retrieval_config` | `ranked_documents` |
| `GenerateAnswerNode` | Rule-based grounded answer assembly from the ranked KB passages only, with numbered citation markers. The caller's free-text query is NEVER rendered into the answer (static lead sentence) — it drives retrieval only | `TrustLevel.ANONYMOUS` | `ranked_documents` | `grounded_answer`, `citations` |
| `OutputFormatNode` | Compose the final answer: body + Sources list + the standing EDU informational disclaimer (disclaimer is composed here; post_process independently enforces it) | `TrustLevel.ANONYMOUS` | `grounded_answer`, `citations` | `formatted_answer`, `status` |

### Data Flow

```
user_input (+ input_context)
  → PreProcessNode (validate + identifier strip)  → validated_input
  → CurriculumKnowledgeGraphNode.extract_input    → stashes input_context (context bridge)
                                                  → inner DomainWorkflowGraph.invoke(validated_input)
        → _extra_initial_state                    → seeds retrieval_config + input_context
        → input_validate                          → search_query / query_filters (fail-closed contract gate)
        → retrieve                                → retrieved_documents
        → rerank_filter                           → ranked_documents
        → generate_answer                         → grounded_answer / citations
        → output_format                           → formatted_answer (+ advisory disclaimer)
     get_output() → {formatted_answer, citations, status, error_log, ...}
  → CurriculumKnowledgeGraphNode.merge_output     → result = formatted_answer, knowledge_base_answer, citations
  → PostProcessNode (output gate + schema)        → formatted_output (gated)
  → AcademicCurriculumKnowledgeAgent.get_output() → base envelope + citations (SUCCESS-only, re-scanned)
```

Structured caller parameters arrive on two channels, validated with the SAME
fail-closed rules by the first inner node (`InputValidateNode`):

1. **`input_context`** (the first-class SDK channel): the HTTP adapter passes
   `{"category": ..., "top_k": ...}` through `agent.invoke(...,
   input_context=...)`. The framework does not forward the outer
   `input_context` into a nested inner graph, so this template bridges it
   explicitly: `extract_input()` stashes it in a `ContextVar`
   (`src/graph/context_bridge.py`) and the inner graph's
   `_extra_initial_state()` reads it back into inner state.
2. **JSON envelope** (in-band legacy form): a caller may supply
   `{"query": ..., "category": ..., "top_k": ...}` as the `input` string; it
   passes through `validated_input` and is parsed back by `InputValidateNode`.

When both channels supply the same field, `input_context` wins — but an
invalid value on EITHER channel rejects the request outright (no silent
fallback to the other channel's value).

### Runtime config forwarding (`_parent_config`)

Runtime parameters live in `config/config.yaml` (the static manifest
`config/agent.yaml` declares identity/entry-point/trust only and carries no
tuning values). The runtime config is resolved by the entry point, not by the graph: each
entry point loads `config/config.yaml` and passes it as `Graph(config=...)`,
so the registry path, the standalone adapter and the Marketplace entry point
all build the graph from the same input. A `BaseNode` has no config
back-reference of its own, so the outer graph reads `self.config` and threads
it into the main-slot node at `register_nodes()` time
(`CurriculumKnowledgeGraphNode(runtime_config=self.config)`).
`_parent_config()` then forwards the `retrieval` + `llm` blocks under
`config["configurable"]` (never `{}` — module fallbacks mirror the file):

```
{"configurable": {"retrieval": {top_k, score_threshold, kb_path}, "llm": {...}}}
```

`get_subgraph()` passes this into `DomainWorkflowGraph(config=...)` (a
graph-level constructor argument — the inner `BaseGraph` ctor is a distinct
contract from the per-node `execute()` signature); the inner graph
republishes the `retrieval` block into the inner initial state as the
JSON-string field `retrieval_config` (via `_extra_initial_state()`), so the
declared `top_k` / `score_threshold` are live at runtime. `RetrieveNode` and
`RerankFilterNode` read `top_k` / `score_threshold` exclusively from the
state-seeded `retrieval_config` field (no `config` parameter reaches node
`execute()` — see "Node execute() contract" below), falling back to safe
module defaults that mirror the config/config.yaml values.

### Node `execute()` contract

Every `FunctionNode` subclass in this template implements EXACTLY
`def execute(self, state) -> dict` — no extra parameters. Config knobs reach
nodes via **State seeding**: `CurriculumKnowledgeGraphNode._parent_config()`
→ inner graph `config` (constructor injection on `DomainWorkflowGraph`, which
is a `BaseGraph`, not a `FunctionNode`) → `DomainWorkflowGraph._extra_initial_state()`
seeds `retrieval_config` into inner State → `RetrieveNode` / `RerankFilterNode`
read the state-seeded key with module-level defaults as fallback. This keeps
every node constructor no-arg, so unit tests construct nodes bare
(`InputValidateNode()`, `RetrieveNode()`, ...).

### State Definition

| Field | Type | Purpose | Layer |
|-------|------|---------|-------|
| `validated_input` | `Optional[str]` | identifier-stripped query payload | outer |
| `knowledge_base_answer` | `Optional[str]` | final answer, mapped from inner `formatted_answer` | outer |
| `search_query` | `Optional[str]` | normalised search query (never rendered into the answer) | inner |
| `query_filters` | `Optional[str]` (JSON) | validated structured params (`category`, `top_k`) | inner |
| `retrieval_config` | `Optional[str]` (JSON) | forwarded runtime `retrieval` block | inner |
| `retrieved_documents` | `Optional[str]` (JSON) | scored KB candidates | inner |
| `ranked_documents` | `Optional[str]` (JSON) | reranked + threshold-filtered passages | inner |
| `grounded_answer` | `Optional[str]` | rule-assembled grounded answer body | inner |
| `citations` | `Optional[str]` (JSON) | `[{ref, id, title, source}]` — re-surfaced at the outer layer by `merge_output()` | both |
| `formatted_answer` | `Optional[str]` | final answer + sources + advisory disclaimer | inner |
| `intake_notes` | `Optional[str]` (JSON) | validation / parse notes (no PII, no echoed values) | inner |
| `trace_id` / `correlation_id` | `Optional[str]` | framework-managed tracing | both |

**State Constraints (mandatory):**
- Flat `TypedDict` only (primitives + JSON-serialisable types).
- Structured fields (dict / list[dict]) stored as JSON STRINGS via `to_json()` /
  `from_json()` — used consistently by every producer AND consumer
  (checkpoint msgpack safety).
- Domain fields are unset at graph initialisation; consumers always read via
  `state.get()`.
- `formatted_output` is NOT re-declared (backbone field stays framework-owned).
- No JWT, API keys, credentials, or raw personal identifiers in State — and no
  student-record fields of any kind (student-record processing is out of
  scope for this template indefinitely).
- `InvocationContext` via `config["configurable"]` only (never in State).
- No Pydantic models / dataclasses / arbitrary Python objects.

## Security Design

- **Trust gate / input validation:** every node declares
  `required_trust_level` (see tables above); `PreProcessNode`
  (VERIFIED_EXTERNAL) rejects empty / non-string `user_input` before the inner
  workflow runs. The standalone server elevates authenticated Bearer callers to
  VERIFIED_EXTERNAL (`INVOKE_AUTH_TOKEN`).
- **Caller-data contract (fail-closed):** every caller-controlled parameter is
  validated against explicit bounds by `InputValidateNode` — `top_k` must be a
  plain integer 1–20 (booleans, floats, numeric strings, and non-finite
  values such as NaN/Infinity are rejected, on both the JSON-envelope and
  `input_context` channels); `category` must be one of the four fixed inert
  category slugs. A violation stops the retrieval path — no search runs and no
  answer is assembled — and the run then completes carrying a reason code, with
  a field-naming message; the rejected value is never echoed into errors,
  notes, or output.

  **Completion is not the same as answering.** A run that ends with
  `AgentStatus.SUCCESS` reports that the request was handled safely to a
  defined end, not that an answer was produced. A value the caller can correct
  (an out-of-contract `top_k` or `category`, an empty question) ends this way
  so that the caller receives the reason and can send a corrected request on
  the same conversation; the alternative — terminating the run — ends the
  calling surface's turn and surfaces only an exception type, leaving the
  reason reachable solely from the audit trail. The reason travels as
  `error_code` in State (`EMPTY_INPUT`, `INVALID_REQUEST`), every later domain
  node passes through without doing work once it is set, and `PostProcessNode`
  renders it as a static caller-facing sentence.

  Two classes keep terminating with ERROR, and must not be folded into the
  above: content refused outright (an instruction-override payload — re-sending
  a reworded variant is not a correction), and a breach of a contract the
  caller cannot influence. Caller strings that could reach rendered output are
  locked to inert identifiers (`[a-z0-9_]{1,32}`): the `channel` label is
  locked by `PreProcessNode`, `category` is enum-validated, and the free-text
  query is never rendered at all. The HTTP adapter additionally caps the
  serialized `input_context` at 256 KiB.
- **Identifier screen:** `PreProcessNode._surface_strip_identifiers()`
  redacts long numeric ID sequences and e-mail patterns from the payload; the
  framework `FunctionNode` default PII scan additionally masks
  `user_input` / `validated_input` at every node boundary.
- **Injection screen (framework):** the framework input gate evaluates
  actionable content for prompt-injection findings before `execute()` runs;
  a high-confidence finding terminates the run with ERROR and no published
  output.
- **Output gate:** `PostProcessNode` calls the module-level
  `_security_gate_output()` scan from `execute()` — API keys / JWT / Bearer
  tokens / credential assignments anywhere in the final `result`, scanned
  RECURSIVELY through nested dict/list/tuple/set structures (never a
  top-level-string-only scan), replace the output with a sanitised stub and
  return `AgentStatus.ERROR`.
  `AcademicCurriculumKnowledgeAgent.get_output()` re-runs the same gate over
  the `citations` structured field before surfacing it (fail-closed twice:
  SUCCESS-only, then re-scanned). No
  `_extra_security_gate_input` / `_extra_security_gate_output` instance
  methods are defined on any node (the framework auto-wraps such hooks —
  prohibited).
- **Output schema enforcement:** the documented contract states that every
  answer carries the standing advisory disclaimer. `OutputFormatNode`
  composes it; `PostProcessNode` independently ENFORCES it — a clean string
  output that lacks the disclaimer is repaired (with an
  `output_disclaimer_enforced` audit event) rather than trusted.
- **Audit logging:** every node's `execute()` emits exactly ONE
  domain-specific `emit_trace_event("<node>_complete", {small non-PII payload},
  state)` (free function, positional args) on its success path. Nodes do NOT
  emit `node_start` / `node_complete` / `node_error` — `BaseNode.__call__()`
  emits those. Domain event names (documented in
  `docs/03_test_spec.md` and the operation guide):
  - `pre_process_complete`
  - `input_validate_complete`
  - `retrieve_complete`
  - `rerank_filter_complete`
  - `generate_answer_complete`
  - `output_format_complete`
  - `post_process_complete`
  - `output_disclaimer_enforced` (only when the schema repair fires)

## Structured Output (`get_output()` override)

The per-claim source citation list IS this template's product differentiator
(a grounded answer with a per-claim source citation), not a side note, so
`AcademicCurriculumKnowledgeAgent.get_output()` EXTENDS
`super().get_output()` (never replaces it — status/node_history/trace_id stay
intact) and adds a decoded `citations` list on top. Fail-closed twice:
(1) only attached when the terminal `state["status"]` is
`AgentStatus.SUCCESS.value`; (2) even on SUCCESS, `citations` is re-scanned
through the same `_security_gate_output()` PostProcessNode uses — a
violation here flips status to ERROR and withholds `citations` rather than
returning a partial payload.

## Advisory Disclaimer

Every answer carries a standing EDU informational disclaimer (the content is
informational only, not an official interpretation of MEXT/NIAD-QE
guidance or institutional policy; verify against the primary regulatory text
and consult the institution's curriculum or accreditation office). It is
composed by `OutputFormatNode` as part of the domain output contract and
independently enforced by `PostProcessNode` (see Security Design).

## Implementation Note — LLM synthesis

The shipped build is **deterministic end-to-end**: retrieval is keyword
scoring over the seeded KB and `GenerateAnswerNode` assembles the grounded
answer rule-based from the ranked passages (static lead sentence + cited
passage excerpts). There is NO live LLM call and no LLM client dependency —
the `llm` block in `config/config.yaml` is forwarded through
`_parent_config()` for forward-compatibility but is not consumed by any
shipped node, and no `system_prompt` is read at runtime. The LLM synthesis
upgrade seam is documented in `config/prompts/answer_synthesis_prompt.md`: an
upgraded `GenerateAnswerNode` swaps the rule-based assembly for an LLM call
that synthesises over the same `ranked_documents` input and emits the same
`grounded_answer` / `citations` state contract, so no other node changes.

## Entry Points

The agent is reachable through three entry points, all of which build the graph
from the same `config/config.yaml`:

| Entry point | Construction | Notes |
|---|---|---|
| Platform registry | `Graph(config=...)` by the registry | Reads `config/config.yaml` itself |
| Standalone HTTP (`src/api/server.py`) | Loads `config/config.yaml`, passes `Graph(config=...)` | Bearer auth boundary; see Security Design |
| Marketplace (`cli.py`) | `run_agent_marketplace(...)` is handed the graph class and the resolved config | The runner constructs the graph itself, so `cli.py` resolves `config/config.yaml` with `load_agent_config()` and passes it in; `extend_config` is the seam for deployment-specific overrides |

`cli.py` sits at the repository root because the deployment image starts it as
`CMD ["python", "cli.py"]`. It adds no business logic: graph construction,
lifecycle, secret provisioning and the invocation loop belong to
`run_agent_marketplace()`.

## Caller-Facing Events

Nodes report progress and rejection reasons to the caller as non-terminal
events, so a caller watching a run sees the pipeline advance instead of a
silent wait, and learns what to change when a request is refused.

- **Progress** — each node reports its phase at the top of `execute()`.
- **Rejection reason** — a node that returns `status: error` sends the reason
  first. It has to happen there: once the run carries an error status the
  framework skips `execute()` on every later node, so no downstream node could
  send it. Wording separates what the caller can fix (missing question,
  oversized request, malformed value) from what they cannot (retrieval or
  output failures), so a caller is not invited into a pointless retry.

Both are best-effort: the emitter is resolved lazily and failures are
swallowed, because reporting must never change the outcome of a run. Messages
are static phase and reason labels — no request value, record value or
internal identifier is ever included, since these events leave the process and
are not covered by the S-3 output gate. Terminal delivery (success/failure)
belongs to the platform runner alone.

## Composition Pattern

- **Pattern:** `GraphNode` (subgraph) in the outer `main` slot.
- **Composition target:** `DomainWorkflowGraph` (inner `BaseGraph`).
- **Error propagation strategy:** `propagate` (inner errors re-raised as
  `SubgraphError`; the framework converts the raised error into a terminal
  ERROR status on the outer state, so a fail-closed inner rejection surfaces
  as `status="error"` with an empty output).
- **Caller-context bridge:** the framework does not forward `input_context`
  on `subgraph.invoke()`, so `extract_input()` stashes it in a `ContextVar`
  and `_extra_initial_state()` seeds it into inner state
  (`src/graph/context_bridge.py`).
- Inner domain nodes run at `TrustLevel.ANONYMOUS`; outer pre/post_process run
  at `TrustLevel.VERIFIED_EXTERNAL`.

## Import Isolation Confirmation
- [x] Template does not import the platform-internal SDK.
- [x] Import targets: `framework/` and `shared/` only.
- [x] No retired intermediate base-class names in any base position.

## Design Decision Record

| Decision | Option A | Option B | Chosen | Rationale |
|----------|----------|----------|--------|-----------|
| Base type | AgentBaseGraph | AutonomousBaseGraph | **AgentBaseGraph** | Fixed multi-step RAG workflow, no autonomous loop |
| Composition pattern | Standalone Cat 1 slots | GraphNode → inner BaseGraph | **GraphNode → inner BaseGraph** | 5-step domain workflow exceeds a single `main` node; nested keeps the outer backbone untouched |
| Answer synthesis | Rule-based assembly | LLM call | **Rule-based (shipped)** | Deterministic assembly is testable offline; an upgrade swaps the LLM in at the documented seam |
| KB storage | External vector store | Seeded JSON KB | **Seeded JSON KB** | Self-contained, deterministic CI; the retrieval contract (`retrieved_documents` JSON) is store-agnostic for a later vector-store upgrade |
| Structured output | Rendered text only | `get_output()` override | **Override — surfaces `citations`** | Per-claim citation is the template's defining feature |
| Caller parameters | In-band JSON envelope only | + first-class `input_context` | **Both, same fail-closed validation** | `input_context` is the platform's structured channel; the envelope stays for in-band callers; one strict contract for both |
