# Answer Synthesis Prompt — EDU-C2-005 (v2 LLM upgrade seam)

> **v1 does NOT use this prompt at runtime.** v1 of `GenerateAnswerNode` is
> deterministic (rule-based grounded assembly over `ranked_documents`); no
> node reads this file. It documents the synthesis contract for the v2 LLM
> upgrade described in `docs/02_design.md` ("v1 Implementation Note — LLM
> synthesis"), so the v2 swap changes only the inside of
> `GenerateAnswerNode.execute()`.

## Contract (v2 GenerateAnswerNode)

- **Input:** the same `ranked_documents` JSON (id / title / category / source /
  score / excerpt) and `search_query` the v1 node reads.
- **Output:** the same state contract — `grounded_answer` (str, with numbered
  `[n]` citation markers) and `citations` (JSON list of
  `{ref, id, title, source}`).
- **Grounding rule:** every factual statement in the answer must be traceable
  to one of the supplied passages via a `[n]` marker; content not present in
  the passages must not be asserted.
- **No-coverage rule:** when no passage supports the question, say so and
  recommend refining the query or escalating to the curriculum/accreditation
  office — never answer from parametric knowledge.
- **Tone:** neutral, administrative/regulatory-appropriate, no individualized
  recommendations (the informational disclaimer is appended downstream by
  `OutputFormatNode`).

## Prompt template

```
You answer questions about MEXT course-of-study guidelines, NIAD-QE
accreditation standards, curriculum requirements, and institutional policy
strictly from the knowledge-base passages provided below.

Question:
{search_query}

Passages (each with a reference number):
{ranked_documents}

Rules:
1. Use ONLY the passages above. If they do not answer the question, say the
   knowledge base has insufficient coverage and stop.
2. Mark every factual statement with the [n] reference of its passage.
3. Do not give an official interpretation or ruling — this is informational
   only, not a substitute for the primary regulatory text.
4. Keep the answer under 300 words.
```

## Manifest coupling

The `llm` block in `config/config.yaml` (`temperature`, `max_tokens`) is already
forwarded to the inner graph via
`CurriculumKnowledgeGraphNode._parent_config()` under
`config["configurable"]["llm"]`; the v2 node reads it from there.
