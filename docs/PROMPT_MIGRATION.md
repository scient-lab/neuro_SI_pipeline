# Prompt Migration Inventory & Plan

**Status (2026-06-20):** 0 of 11 production prompts currently consume YAML.
4 YAML files exist under `prompts/` but are orphaned — `pipeline_config.get_prompt()`
is defined but never called by any consumer code. Every LLM call in the pipeline
runs against a hardcoded Python string.

This document inventories every prompt source, maps the current state to a
target YAML location, and tracks migration progress.

> **Related**: `.claude/docs/ARCHITECTURE_5_REPO_DECISION.md`,
> `ORCHESTRATION_PLAN.md`, `feedback_no_hardcoding` memory rule.

---

## 1. Headline

| Metric | Current | Target |
|---|---|---|
| Prompts consumed from YAML at runtime | **0** | 11 |
| Hardcoded Python prompts in active use | **11** | 0 |
| Orphaned YAML files (defined, never read) | 4 | 0 |
| Phases reading from `domains/<name>.yaml` for content | 0 (none use `get_prompt`) | 11 |

The `prompts/extract.yaml`, `prompts/validate.yaml`, `prompts/curriculum_check.yaml`,
and `prompts/curriculum_qa.yaml` files exist as scaffolding from an aborted
earlier migration. The real prompts that run at LLM-call time live in the
`*_prompts.py` files and inline `f"""..."""` strings listed below.

---

## 2. Inventory — all prompts found

Status legend:
- **✅** YAML-driven via `get_prompt()` (none yet)
- **🟡** YAML file exists but is orphaned (nobody reads it)
- **❌** Hardcoded in Python, no YAML target yet
- **🚨** Hardcoded AND domain content is wrong (e.g. diabetes content in a
        neuroscience pipeline) — these are bugs

| # | Status | Phase / Step | Current source | Target YAML | Domain | LOC | TODO |
|---|---|---|---|---|---|---|---|
| 1 | 🟡❌ | `extract` (graphrag) | `1_seed_kg/prompts_kg.py:54-127` (`PROMPT_TEMPLATE` + `ASSISTANT_EXAMPLE`) | `prompts/extract.yaml` (exists, orphaned) | neuroscience (inline) | ~70 | ☐ wire `prompts_kg.py` to `get_prompt("extract")` ☐ move neuroscience-specific content to `domains/neuroscience.yaml::extract_examples` |
| 2 | 🚨 | `graphmert / entity_discovery` | `2_graphmert/utils/entity_discovery/entity_discovery_prompts.py:1` (`SYSTEM_CONTEXT`) | `prompts/entity_discovery.yaml` (new) | **diabetes — WRONG for our pipeline** | 242 | ☐ create template ☐ wire `entity_discovery.py:34` to `get_prompt("entity_discovery")` ☐ verify content sourced from `domains/neuroscience.yaml` |
| 3 | ❌ | `graphmert / entity_discovery` | `2_graphmert/utils/entity_discovery/entity_discovery.py:36` (`USER_TEMPLATE`) | (folds into `prompts/entity_discovery.yaml::user`) | format only | 1 | ☐ inline into the entity_discovery YAML user block |
| 4 | ❌ | `graphmert / add_llm_relations` | `2_graphmert/utils/relation_matching/relation_match_prompts.py:167-225` (`SYSTEM_CONTEXT_TEMPLATE` + `EXAMPLES_*` + `MEANING_EXPL_*`) | `prompts/add_llm_relations.yaml` (new) | neuroscience (already adapted) | 242 | ☐ create template ☐ wire `add_llm_relations.py:36` ☐ move neuroscience examples to `domains/neuroscience.yaml::relation_match_examples` |
| 5 | ❌ | `graphmert / combine_tails` | `2_graphmert/utils/combine_tails/combine_tokens_prompts.py:1-67` (`MEANING_EXPL_NEURO` + `SYSTEM_CONTEXT_TEMPLATE`) | `prompts/combine_tails.yaml` (new) | neuroscience (already adapted) | ~150 | ☐ create template ☐ wire `combine_tokens.py` (consumer) ☐ move content to `domains/neuroscience.yaml::combine_examples` |
| 6 | ❌ | `graphmert / predict_tails` | `2_graphmert/predict_tails_llm.py:72` (`SYSTEM_PROMPT`) | `prompts/predict_tails.yaml` (new) | neuroscience (inline) | ~30 | ☐ create template ☐ wire predict_tails_llm.py |
| 7 | 🟡❌ | `validate` (2-LLM consensus) | (no consumer found — `prompts/validate.yaml` exists but unused) | `prompts/validate.yaml` (exists, orphaned) | neuroscience | n/a | ☐ identify the actual validate consumer ☐ wire it to `get_prompt("validate")` |
| 8 | 🟡❌ | `curriculum_check` | (no consumer — orphaned YAML) | `prompts/curriculum_check.yaml` (exists, orphaned) | neuroscience | n/a | ☐ identify consumer ☐ wire to YAML |
| 9 | 🟡❌ | `curriculum_qa` | (no consumer — orphaned YAML) | `prompts/curriculum_qa.yaml` (exists, orphaned) | neuroscience | n/a | ☐ identify consumer ☐ wire to YAML |
| 10 | ❌ | `curriculum / generate_questions` | `3_si_curriculum/curriculum_generator/generate_questions.py:284, 339, 443, 473, 502, 551` (inline `f"""..."""` × 6) | `prompts/curriculum_generate.yaml` (new, multi-sub-prompt) | neuroscience (inline) | ~150 total | ☐ extract each inline f-string ☐ create YAML with sub-keys per generation step ☐ wire `generate_questions.py` to `get_prompt("curriculum_generate")` |
| 11 | ❌ | `curriculum / verify_questions` | `3_si_curriculum/curriculum_generator/verify_questions.py:36` (`SYSTEM_PROMPT_QA_VALIDATION`) | `prompts/curriculum_verify.yaml` (new) | neuroscience (inline) | ~30 | ☐ create template ☐ wire consumer |
| 12 | ❌ | `curriculum / test_models / eval` | `3_si_curriculum/test_models/eval_models.py:34, 41` (`SYSTEM_PROMPT`, `GEMINI_SYSTEM_PROMPT`) | `prompts/eval_models.yaml` (new) | generic MCQ instructions | ~20 | ☐ create template ☐ wire eval_models.py |
| 13 | ❌ | `rl / training` | `3_si_curriculum/RL/rl_training.py:79` (`SYSTEM_PROMPT`) | `prompts/rl_training.yaml` (new) | generic MCQ instructions | ~10 | ☐ create template ☐ wire consumer |
| 14 | ❌ | `rl / test` | `3_si_curriculum/RL/test_rl.py:43` (`SYSTEM_PROMPT`) | `prompts/rl_test.yaml` (new — or share `rl_training.yaml`) | generic MCQ instructions | ~10 | ☐ create template ☐ wire consumer |

**Totals**: 14 prompt sources × 11 distinct phases. Includes 2 cosmetic items
(USER_TEMPLATE format string, RL test/train share same content).

---

## 3. Per-prompt detail

### 3.1 #1 — `extract` (graphrag)

**Status:** 🟡❌ — YAML exists (`prompts/extract.yaml`) but the actual consumer
ignores it and uses the hardcoded `PROMPT_TEMPLATE` inside `prompts_kg.py`.

<table>
<tr>
<th width="50%">Current — <code>1_seed_kg/prompts_kg.py:54-127</code></th>
<th width="50%">Target — <code>prompts/extract.yaml</code> (already exists, needs wiring)</th>
</tr>
<tr>
<td>

```python
PROMPT_TEMPLATE = """-Role-
You are an AI assistant specialized in
extracting structured information from
neuroscience textbook content to build a
knowledge graph about the nervous system,
brain function, and neural mechanisms.

-Goal-
Given neuroscience textbook content, a
predefined list of entity types, and a
predefined list of relations, identify
EVERY SINGLE entity of those types and
the scientifically meaningful relationships
explicitly described among the identified
entities within the text.
...
[70+ lines of inline neuroscience
 instructions + entity types + examples]
"""

ASSISTANT_EXAMPLE = """("entity"
{tuple_delimiter}voltage-gated sodium
channels{tuple_delimiter}Molecular Entity
{tuple_delimiter}ion channels responsible
for the rising phase of the action
potential, ...
"""
```

</td>
<td>

```yaml
name: extract
phase: extract
system: |
  You are a {{domain}} domain expert
  extracting structured knowledge from text.
  Use chain-of-thought reasoning internally.
  Output ONLY valid JSON in the required
  schema. Do not include explanations
  outside the JSON.
user: |
  {{focus_instructions}}
  Allowed entity categories:
  {{categories}}
  Allowed relations:
  {{relations}}
  Examples:
  {{few_shot}}
  Text to extract from:
  {{text}}
generation:
  temperature: 0.0
  max_tokens: 2048
  response_format: json
```

</td>
</tr>
</table>

**Slots filled from `domains/neuroscience.yaml`**: `domain`, `focus_instructions`,
`entity_categories`, `relations`, `few_shot_examples`.

**TODO:**
- [ ] Replace `PROMPT_TEMPLATE` consumer in `prompts_kg.py` with a renderer
      that calls `get_prompt("extract")` and substitutes the slots.
- [ ] Move the inline neuroscience-specific exemplar text (sodium channels
      example) from `prompts_kg.py:108-127` into `domains/neuroscience.yaml`
      as a structured `extract_examples:` block.
- [ ] Delete `PROMPT_TEMPLATE` and `ASSISTANT_EXAMPLE` constants from
      `prompts_kg.py` (keep the helper functions only).
- [ ] Validate graphrag extracts at the same triple count/quality after the
      swap (snapshot kg_final.csv before/after).

---

### 3.2 #2 — `entity_discovery` 🚨 **(blocking smoke run)**

**Status:** 🚨 — hardcoded prompt has DIABETES content; pipeline is
neuroscience. This is the bug causing the current smoke failure (0 grounded
triples → Keys mismatch error in graphmert.preprocess).

<table>
<tr>
<th width="50%">Current — <code>entity_discovery_prompts.py:1-242</code></th>
<th width="50%">Target — <code>prompts/entity_discovery.yaml</code> (new)</th>
</tr>
<tr>
<td>

```python
SYSTEM_CONTEXT = """You are a medical-
domain extractor building a diabetes KG
of <head, relation, tail>. You possess
advanced medical academic knowledge.

Given input sequence, identify entities
specifically relevant to diabetes, its
complications, comorbidities,
therapeutics and related biomedical
entities that help to clarify or
contextualize them. Output a Python list
of up to 6-word entity "heads" following
these rules:

  1. Select a precise and medically-
     specific span (e.g., "myocardial
     infarction," not "infarction"). ...
  2. Keep original spelling, casing, and
     abbreviations from the sequence.
  3. Do not include COVID-related terms.
     Do not include head entities that
     describe findings in animal models
     (mice, rats, etc.).
...
[242 lines of diabetes-specific rules,
 examples like beta cells of the pancreas,
 nhanes 2015-2018, β-cell death, etc.]
"""
```

Consumed at `entity_discovery.py:34`:

```python
from entity_discovery_prompts \
    import SYSTEM_CONTEXT as SYSTEM_PROMPT
```

And user template at `entity_discovery.py:36`:

```python
USER_TEMPLATE = "Input:\n{text}\n\nOutput:"
```

</td>
<td>

```yaml
name: entity_discovery
phase: graphmert.preprocess.entity_discovery
system: |
  You are a {{domain}} domain expert
  extracting entity heads from text. Each
  head is a noun referring to a specific
  concept of interest to a {{domain}}
  knowledge graph.

  Output format: a Python list of strings,
  e.g. ['head_1', 'head_2', ...]. Wrap
  intermediate reasoning in <think>...
  </think>; the final list comes AFTER
  </think>.

  Rules:
  - Each head must be LOWERCASE and
    CANONICAL (strip articles, trailing
    punctuation, plural marker if a
    singular form is more canonical).
  - Each head must fit one of the allowed
    entity categories.
  - Discard generic / vague terms.
  - Discard concepts that do not
    contribute meaningful {{domain}}
    knowledge.

user: |
  {{focus_instructions}}

  Allowed entity categories (emit only
  entities of these types):
  {{categories}}

  Input:
  {{text}}

  Output:

generation:
  temperature: 0.0
  max_tokens: 512
```

</td>
</tr>
</table>

**Slots filled from `domains/neuroscience.yaml`**: `domain`, `focus_instructions`,
`entity_categories`. No `few_shot` slot — entity_categories alone is enough
guidance.

**Key diffs**:
- "diabetes KG" → "{{domain}} knowledge graph"
- "Keep original spelling, casing" → "LOWERCASE and CANONICAL" (fixes the
  trailing-comma bug we hit)
- Diabetes-specific exclusions (COVID, animal models) → dropped
- 242 lines of inline content → ~30 lines of template + content in domain YAML

**TODO:**
- [ ] Create `prompts/entity_discovery.yaml`.
- [ ] Patch `entity_discovery.py:34` to import + call `get_prompt(...)` and
      render with a tiny `_render(prompt_dict, **slots)` helper (Python
      `str.replace("{{slot}}", str(value))`).
- [ ] Remove `from entity_discovery_prompts import ...` line.
- [ ] Remove the `USER_TEMPLATE = "Input:\n{text}\n\nOutput:"` constant at
      `entity_discovery.py:36`.
- [ ] Leave `entity_discovery_prompts.py` for one commit (orphaned); delete
      in a follow-up commit so `git bisect` stays clean.
- [ ] Re-run smoke graphmert.preprocess and verify >0 grounded triples in
      `Grounding results: ...` log line.

---

### 3.3 #4 — `add_llm_relations`

**Status:** ❌ neuroscience-correct content, but hardcoded.

<table>
<tr>
<th width="50%">Current — <code>relation_match_prompts.py</code></th>
<th width="50%">Target — <code>prompts/add_llm_relations.yaml</code> (new)</th>
</tr>
<tr>
<td>

```python
EXAMPLES_gemini_score45 = """
hippocampus | part_of | limbic system
microglia | participates_in | neuroinflammation
serotonin | modulates | mood regulation
cerebral cortex | contains | pyramidal neurons
substantia nigra pars compacta | located_in | midbrain
primary motor cortex | projects_to | spinal cord
...
[~75 hardcoded neuroscience triple examples]
"""

MEANING_EXPL_qwen32_score55_exp_kg = """
These relations are neuroscience KG relations.
[~80 lines explaining each relation semantics
 — projects_to, modulates, mediates_signal_for,
 binds_to, expressed_in, etc.]
"""

SYSTEM_CONTEXT_TEMPLATE = """You are a
neuroscience-domain extractor building a
neuroscience knowledge graph (KG) of triples
<head, relation, tail>.
{MEANING}
Examples:
{EXAMPLES}
...
"""

SYSTEM_CONTEXT = SYSTEM_CONTEXT_TEMPLATE.format(
    EXAMPLES=EXAMPLES_qwen32_score55_exp_kg,
    MEANING=MEANING_EXPL_qwen32_score55_exp_kg,
    ...)
```

</td>
<td>

```yaml
name: add_llm_relations
phase: graphmert.preprocess.add_llm_relations
system: |
  You are a {{domain}} domain extractor
  identifying which relations apply to each
  candidate (head, *, ?) pair given input text.

  Relation meanings:
  {{relation_meanings}}

  Examples:
  {{relation_examples}}

  Rules:
  - Use only relations from the allowed list.
  - Use direction exactly as written.
  - Output JSON: {"head1": ["rel1"], ...}.

user: |
  Input:
  {{text}}
  Candidate heads: {{heads}}
  Output:

generation:
  temperature: 0.0
  max_tokens: 1024
```

**New domain slots** to add to
`domains/neuroscience.yaml`:

```yaml
# from MEANING_EXPL_qwen32_score55_exp_kg
relation_meanings: |
  projects_to: anatomical projection of
    neuron / pathway to target region
  modulates: alters activity without
    primary causation
  ...

# from EXAMPLES_gemini_score45
relation_examples: |
  hippocampus | part_of | limbic system
  microglia | participates_in | ...
  ...
```

</td>
</tr>
</table>

**TODO:**
- [ ] Create `prompts/add_llm_relations.yaml`.
- [ ] Move `MEANING_EXPL_qwen32_score55_exp_kg` to
      `domains/neuroscience.yaml::relation_meanings`.
- [ ] Move `EXAMPLES_gemini_score45` to
      `domains/neuroscience.yaml::relation_examples`.
- [ ] Add `pipeline_config.get_relation_meanings()` and
      `get_relation_examples()` helpers (mirror pattern of existing
      `get_relations()`, `get_few_shot_examples()`).
- [ ] Wire `add_llm_relations.py` to use `get_prompt("add_llm_relations")`.
- [ ] Delete the `*_prompts.py` constants.

---

### 3.4 #5 — `combine_tails`

**Status:** ❌ neuroscience-correct content, hardcoded.

<table>
<tr>
<th width="50%">Current — <code>combine_tokens_prompts.py:1-67</code></th>
<th width="50%">Target — <code>prompts/combine_tails.yaml</code> (new)</th>
</tr>
<tr>
<td>

```python
MEANING_EXPL_NEURO = """Relation meanings
(neuroscience KG). Use the direction exactly
as written.
[~40 lines of relation semantics]
"""

SYSTEM_CONTEXT_TEMPLATE = """You are an
expert Neuroscience Knowledge Graph curator.
You specialize in identifying highly specific
biological entities.
...
{MEANING_EXPL}
..."""

SYSTEM_CONTEXT = SYSTEM_CONTEXT_TEMPLATE.format(
    MEANING_EXPL=MEANING_EXPL_NEURO,
)
```

</td>
<td>

```yaml
name: combine_tails
phase: graphmert.predict_tails.combine
system: |
  You are an expert {{domain}} knowledge
  graph curator. You specialize in
  identifying highly specific biological
  entities.

  Relation meanings:
  {{relation_meanings}}

  Rules:
  - Use direction exactly as written.
  - Output JSON-encoded merged tail token
    sequence per (head, relation) pair.

user: |
  Head: {{head}}
  Relation: {{relation}}
  Candidate tail token sequences:
  {{candidates}}

  Output:

generation:
  temperature: 0.0
  max_tokens: 512
```

**Reuses** `{{relation_meanings}}` slot
already added by #4 — single source of
truth in `domains/neuroscience.yaml`.

</td>
</tr>
</table>

**TODO:**
- [ ] Create `prompts/combine_tails.yaml`.
- [ ] Reuse `domains/neuroscience.yaml::relation_meanings` (avoid duplication
      with #4).
- [ ] Wire `combine_tails.py` consumer.

---

### 3.5 #6 — `predict_tails`

**Status:** ❌ neuroscience content (inline), hardcoded.

<table>
<tr>
<th width="50%">Current — <code>predict_tails_llm.py:72</code></th>
<th width="50%">Target — <code>prompts/predict_tails.yaml</code> (new)</th>
</tr>
<tr>
<td>

```python
SYSTEM_PROMPT = """
You are a neuroscience-domain expert
ranking candidate tail entities for a
given (head, relation, ?) query.

Given a source snippet that mentions a
head entity, a query relation, and a list
of candidate tail entities, return the
top-K candidates ranked by likelihood the
relation holds between the head and the
candidate tail.

Be strict: only rank tails that are
explicitly supported by the snippet text.
Discard candidates that require inference
beyond the snippet.

Output: JSON array of candidate strings
in decreasing order of confidence.
"""
```

</td>
<td>

```yaml
name: predict_tails
phase: graphmert.predict_tails
system: |
  You are a {{domain}} domain expert
  ranking candidate tail entities for
  (head, relation, ?) queries given source
  text snippets.

  Be strict: only rank tails that are
  explicitly supported by the snippet text.
  Discard candidates that require
  inference beyond the snippet.

  Output: JSON array of candidate strings
  in decreasing order of confidence.

user: |
  Source snippet: {{snippet}}
  Head: {{head}}
  Relation: {{relation}}
  Candidate tails: {{candidates}}
  Output:

generation:
  temperature: 0.0
  max_tokens: 256
```

</td>
</tr>
</table>

**TODO:**
- [ ] Create `prompts/predict_tails.yaml`.
- [ ] Wire `predict_tails_llm.py`.

---

### 3.6 #7, #8, #9 — orphaned YAMLs (validate, curriculum_check, curriculum_qa)

**Status:** 🟡❌ — YAML files exist but no consumer code calls `get_prompt()`
for them. Need to audit which Python file SHOULD have been wired.

<table>
<tr>
<th width="50%">Current — orphaned YAMLs</th>
<th width="50%">Target — find + wire consumer</th>
</tr>
<tr>
<td>

```text
prompts/validate.yaml         ← EXISTS but nobody reads it
prompts/curriculum_check.yaml ← EXISTS but nobody reads it
prompts/curriculum_qa.yaml    ← EXISTS but nobody reads it

# Each likely has a Python sibling that
# was meant to read it but never did,
# OR is being read via a hardcoded
# string somewhere we haven't found.

grep -rn 'validate' --include='*.py' .
grep -rn 'curriculum_check' --include='*.py' .
grep -rn 'curriculum_qa' --include='*.py' .
```

</td>
<td>

```python
# For each orphan:
# 1. Identify the consumer Python file
#    (likely some *_check.py or
#    validate_*.py)
# 2. Check whether it has a hardcoded
#    SYSTEM_PROMPT
# 3. Replace with:
#
from pipeline_config import render_prompt
prompt = render_prompt("validate",
                       text=input_text)
response = llm.chat([
    {"role": "system",
     "content": prompt["system"]},
    {"role": "user",
     "content": prompt["user"]},
])
```

</td>
</tr>
</table>

**TODO for each:**
- [ ] **#7 validate**: grep for validate phase LLM calls; wire to
      `get_prompt("validate")`.
- [ ] **#8 curriculum_check**: same audit.
- [ ] **#9 curriculum_qa**: same audit.
- [ ] If a consumer is hardcoded with a similar prompt body, replace with
      `get_prompt(...)` call AND remove the hardcoded string.

---

### 3.7 #10 — `curriculum / generate_questions`

**Status:** ❌ 6 separate inline `f"""..."""` prompts in one file.

<table>
<tr>
<th width="50%">Current — <code>generate_questions.py</code> (6 sites)</th>
<th width="50%">Target — <code>prompts/curriculum_generate.yaml</code> (multi-section)</th>
</tr>
<tr>
<td>

```python
# Line 284 — KG path filter
prompt = f"""You are a strict evaluator
deciding whether a knowledge graph path
can produce a genuinely challenging
multi-hop neuroscience exam question. Be
AGGRESSIVE about skipping ..."""

# Line 339 — primary question generation
prompt = f"""
[question generation instructions]
"""

# Line 443 — alternative question generation
prompt = f"""[...]"""

# Line 473 — reasoning trace
prompt = f"""Generate a thinking trace for
the following neuroscience question. The
correct answer is {correct_answer}. ..."""

# Line 502 — trace tightening
stricter_prompt = f"""You previously
generated a {word_count}-word explanation.
That is too long. Rewrite it in EXACTLY
{TRACE_TARGET_WORDS} words or fewer. ..."""

# Line 551 — final verification
prompt = f"""You are a neuroscience
examiner. You are given a neuroscience
question and an explanation with an
answer. ..."""
```

</td>
<td>

```yaml
name: curriculum_generate
phase: curriculum.generate_questions

prompts:
  path_filter:
    system: |
      You are a strict evaluator deciding
      whether a knowledge graph path can
      produce a challenging multi-hop
      {{domain}} exam question. Be
      AGGRESSIVE about skipping borderline
      paths.
    user: ...

  generate:
    system: ...
    user: ...

  generate_alt:
    system: ...
    user: ...

  trace:
    system: ...
    user: |
      Generate a thinking trace for the
      following {{domain}} question.
      Correct answer: {{correct_answer}}.

  trace_tighten:
    system: ...
    user: |
      You previously generated a
      {{word_count}}-word explanation.
      That is too long. Rewrite it in
      EXACTLY {{trace_target_words}}
      words or fewer. ...

  verify:
    system: |
      You are a {{domain}} examiner. ...
    user: ...

generation:
  temperature: 0.1
  max_tokens: 1024
```

**Constants migrated to YAML:**
- `TRACE_TARGET_WORDS` →
  `configs/default.yaml::curriculum.trace_target_words`

</td>
</tr>
</table>

**TODO:**
- [ ] Extract all 6 inline f-strings.
- [ ] Restructure as a multi-section YAML (each sub-prompt as a key under
      `prompts:`).
- [ ] Update `generate_questions.py` to call `get_prompt("curriculum_generate")`
      and access `["prompts"]["path_filter"]`, etc.
- [ ] Move `TRACE_TARGET_WORDS` constant to `configs/default.yaml::curriculum`
      or `domains/neuroscience.yaml`.

---

### 3.8 #11 — `curriculum / verify_questions`

**Status:** ❌ neuroscience content (inline), hardcoded.

<table>
<tr>
<th width="50%">Current — <code>verify_questions.py:36</code></th>
<th width="50%">Target — <code>prompts/curriculum_verify.yaml</code> (new)</th>
</tr>
<tr>
<td>

```python
SYSTEM_PROMPT_QA_VALIDATION = """You are
an editor for a graduate-level
neuroscience exam dataset.

Given a candidate (question, answer,
reasoning trace) tuple, evaluate whether
the question is suitable for a
{exam_grade}-level audience and whether
the reasoning supports the stated answer.

Reject items that:
- Have ambiguous wording
- Lack neuroscience specificity
- Have a reasoning trace that does not
  support the answer

Output JSON: {"keep": bool, "reason": str}
"""
```

</td>
<td>

```yaml
name: curriculum_verify
phase: curriculum.verify_questions
system: |
  You are an editor for a graduate-level
  {{domain}} exam dataset.

  Given a candidate (question, answer,
  reasoning trace) tuple, evaluate whether
  the question is suitable for a
  {{exam_grade}}-level audience and whether
  the reasoning supports the stated
  answer.

  Reject items that:
  - Have ambiguous wording
  - Lack {{domain}} specificity
  - Have a reasoning trace that does not
    support the answer

  Output JSON: {"keep": bool, "reason": str}

user: |
  Question: {{question}}
  Answer: {{answer}}
  Reasoning trace: {{trace}}

generation:
  temperature: 0.0
  max_tokens: 256
```

</td>
</tr>
</table>

**TODO:**
- [ ] Create `prompts/curriculum_verify.yaml`.
- [ ] Wire `verify_questions.py`.

---

### 3.9 #12 — `test_models / eval_models`

**Status:** ❌ generic MCQ prompts (not domain-specific), hardcoded.

<table>
<tr>
<th width="50%">Current — <code>eval_models.py:34, 41</code></th>
<th width="50%">Target — <code>prompts/eval_models.yaml</code> (new)</th>
</tr>
<tr>
<td>

```python
SYSTEM_PROMPT = (
    "A conversation between user and "
    "assistant. The user asks a single-"
    "choice Multiple Choice Question, and "
    "the assistant solves it using step-"
    "by-step reasoning. Please answer the "
    "multiple choice question by selecting "
    "only one from option A, option B, "
    "option C, option D."
)

GEMINI_SYSTEM_PROMPT = (
    "..."
    " (gemini-specific phrasing variant)"
)
```

</td>
<td>

```yaml
name: eval_models
phase: curriculum.eval

prompts:
  default:
    system: |
      A conversation between user and
      assistant. The user asks a
      single-choice Multiple Choice
      Question, and the assistant solves
      it using step-by-step reasoning.
      Please answer by selecting only one
      of A/B/C/D.

  gemini:
    system: |
      ... (gemini-specific phrasing)

generation:
  temperature: 0.0
```

</td>
</tr>
</table>

**TODO:**
- [ ] Create `prompts/eval_models.yaml`.
- [ ] Wire `eval_models.py`.

---

### 3.10 #13, #14 — RL training + RL test

**Status:** ❌ identical MCQ prompts duplicated across two files.

<table>
<tr>
<th width="50%">Current — <code>rl_training.py:79</code> AND <code>test_rl.py:43</code></th>
<th width="50%">Target — <code>prompts/rl_mcq.yaml</code> (new; shared)</th>
</tr>
<tr>
<td>

```python
# IDENTICAL string in both files:

SYSTEM_PROMPT = """A conversation between
user and assistant. The user asks a
single-choice Multiple Choice Question,
and the assistant solves it using
step-by-step reasoning. Please answer the
multiple choice question by selecting only
one from option A, option B, option C,
option D.
"""
```

**Duplicate**: same string in both
`rl_training.py:79` and `test_rl.py:43`.
Migration consolidates to one file.

</td>
<td>

```yaml
name: rl_mcq
phase: rl
system: |
  A conversation between user and
  assistant. The user asks a single-choice
  Multiple Choice Question, and the
  assistant solves it using step-by-step
  reasoning. Please answer by selecting
  only one of A/B/C/D.

generation:
  temperature: 0.7
```

**Wire BOTH** `rl_training.py` and
`test_rl.py` to read from this single
file via `get_prompt("rl_mcq")`.
Eliminates duplication.

</td>
</tr>
</table>

**TODO:**
- [ ] Create `prompts/rl_mcq.yaml`.
- [ ] Wire both `rl_training.py` AND `test_rl.py` to read from the same YAML
      (eliminates the duplication).

---

## 4. Render pipeline (needs implementation)

`pipeline_config.get_prompt(name)` currently returns the **raw dict**. It does
NOT perform slot substitution. Callers must render templates themselves.

<table>
<tr>
<th width="50%">Current — <code>pipeline_config.py:176</code></th>
<th width="50%">Target — add <code>render_prompt()</code> helper</th>
</tr>
<tr>
<td>

```python
def get_prompt(name: str) -> dict[str, Any]:
    """Return the prompt template for a phase
    by name.

    Lookup order (first hit wins):
      1. prompts/overrides/<SI_DOMAIN>/<name>.yaml
      2. prompts/<name>.yaml
    """
    domain = get_domain_name()
    override = _REPO_ROOT / "prompts" \
        / "overrides" / domain / f"{name}.yaml"
    if override.is_file():
        return _read(override)
    return _read(_REPO_ROOT / "prompts"
                 / f"{name}.yaml")

# Returns the raw YAML dict.
# NO {{slot}} substitution.
# NO auto-fill from domains/<name>.yaml.
# Callers would have to do that themselves
# — which is why nothing in the codebase
# uses get_prompt() today.
```

</td>
<td>

```python
def render_prompt(name: str, **slots) -> dict:
    """Load prompts/<name>.yaml, substitute
    {{slot}} placeholders, return rendered
    system + user messages.

    Auto-fills slots from active SI_DOMAIN
    unless passed explicitly:
      - domain
      - focus_instructions
      - categories
      - relations
      - few_shot
      - relation_meanings
      - relation_examples

    Returns:
      {"system": "...", "user": "...",
       "generation": {...}}
    """
    prompt = get_prompt(name)
    defaults = {
      "domain":   get_domain_name(),
      "focus_instructions":
        get_focus_instructions(),
      "categories":
        _format_list(get_entity_categories()),
      "relations":
        _format_list(get_relations()),
      "few_shot":
        _format_examples(
          get_few_shot_examples()),
    }
    slots = {**defaults, **slots}
    return {
      "system":
        _substitute(prompt.get("system",""),
                    slots),
      "user":
        _substitute(prompt.get("user",""),
                    slots),
      "generation":
        prompt.get("generation", {}),
    }

def _substitute(text: str, slots: dict) -> str:
    for k, v in slots.items():
        text = text.replace(
            f"{{{{{k}}}}}", str(v))
    return text
```

</td>
</tr>
</table>

**TODO:**
- [ ] Add `render_prompt()` to `pipeline_config.py`.
- [ ] Add `get_relation_meanings()`, `get_relation_examples()` helpers as
      needed by individual phase migrations.
- [ ] Document expected slot inventory per prompt in each YAML's
      top-of-file comment.

---

## 5. Migration sequence

Priority ordered by (impact × ease). Higher priority = do first.

| Order | Item | Priority | Reason |
|---|---|---|---|
| 1 | #2 entity_discovery | **CRITICAL** | Blocking smoke run RIGHT NOW |
| 2 | Render helper in pipeline_config | **CRITICAL** | Foundation for all subsequent migrations |
| 3 | #1 extract (wire orphaned YAML) | High | Validates the render pipeline against the most complex prompt |
| 4 | #7-#9 orphaned validate/curriculum YAMLs | Medium | Already-written YAMLs; just need to find consumers |
| 5 | #4 add_llm_relations | Medium | Big content move; reduces graphmert phase's hardcoded surface area significantly |
| 6 | #6 predict_tails | Medium | Small file; quick win |
| 7 | #5 combine_tails | Medium | Reuses #4's `relation_meanings` slot |
| 8 | #13-#14 RL prompts (consolidate) | Low | Generic; eliminates duplicate |
| 9 | #12 eval_models | Low | Generic |
| 10 | #11 curriculum_verify | Low | Single prompt |
| 11 | #10 curriculum_generate | Low | Most complex (6 sub-prompts); leave for last when pattern is validated |

---

## 6. Audit / regression prevention

After migration, add to `scripts/diagnose.sh` (or as a new section in the
upcoming `scripts/analysis.sh`):

```bash
# §X. Hardcoded prompt audit
section "X. Hardcoded prompt audit"

# Find any new SYSTEM_PROMPT / PROMPT_TEMPLATE / inline f""" prompt strings
hardcoded=$(grep -rnE '(SYSTEM_PROMPT|SYSTEM_CONTEXT|PROMPT_TEMPLATE)\s*=\s*"""' \
    --include='*.py' \
    1_seed_kg/ 2_graphmert/ 3_si_curriculum/ \
    | grep -v __pycache__ | wc -l)

if [[ "$hardcoded" -gt 0 ]]; then
    mark_fail "$hardcoded hardcoded prompt strings found"
    grep -rnE '(SYSTEM_PROMPT|SYSTEM_CONTEXT|PROMPT_TEMPLATE)\s*=\s*"""' \
        --include='*.py' 1_seed_kg/ 2_graphmert/ 3_si_curriculum/ \
        | grep -v __pycache__ | head -10 | sed 's/^/    /'
else
    mark_ok "no hardcoded prompts detected"
fi

# Cross-check: every prompts/*.yaml should have at least one consumer
for yml in prompts/*.yaml; do
    name=$(basename "$yml" .yaml)
    consumers=$(grep -rln "get_prompt(\"$name\"" --include='*.py' . | wc -l)
    if [[ "$consumers" -eq 0 ]]; then
        mark_warn "prompts/$name.yaml has no consumer (orphaned)"
    fi
done

# Cross-check: any 'diabetes' in non-biomed pipelines
if [[ "${SI_DOMAIN:-neuroscience}" != "biomed" ]]; then
    diabetes_hits=$(grep -riE '\bdiabetes\b' --include='*.py' --include='*.yaml' \
        2_graphmert/ 3_si_curriculum/ prompts/ domains/ \
        | grep -v __pycache__ | grep -v 'allowed_off_domain' | wc -l)
    if [[ "$diabetes_hits" -gt 0 ]]; then
        mark_fail "domain leak: $diabetes_hits 'diabetes' mentions in non-biomed pipeline"
    fi
fi
```

**TODO:**
- [ ] Add the §X section to `scripts/diagnose.sh` (or `scripts/analysis.sh`).
- [ ] Add `diagnose.sh` invocation to pre-merge CI / pre-commit hook.

---

## 7. Status changelog

- **2026-06-20** — Initial inventory written.
  - 14 prompt sources identified across 11 phases.
  - 4 orphaned YAML files documented.
  - 0 prompts currently consume YAML.
  - #2 entity_discovery flagged as smoke-blocker.
