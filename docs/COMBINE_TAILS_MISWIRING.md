# combine_tails is miswired as a filter — it should MERGE + DEDUPLICATE

**Date:** 2026-07-01
**Status:** ✅ RESOLVED 2026-07-02 — Jake Stephen fixed it upstream in **`dc5bb46`** ("Fix combine_tails to merge/dedup only; fact_score is the sole LLM filter"), confirming this diagnosis. Ported to `orchestration` 2026-07-02: combine_tails.py code-copied from dc5bb46 (merge + `drop_duplicates`, writes `final_kg_combined.csv`, no LLM); `graphmert.sh` combine_tails call stripped of `--model_id`/`--tokenizer` and fact_score input renamed; dead knobs removed from `default.yaml` (combine_tails_max_model_len/tensor_parallel_size/max_tokens/no_think); `prompts/combine_tails.yaml` + `combine_tokens_prompts.py` deleted. fact_score-twice (seed + expansion) was already wired. **Still open (separate):** the `predict_tails_gm` parquet→shard bridge (GraphMERT-MLM tails still don't reach combine_tails — see graphmert.sh integration-gap warning).
**Scope:** `2_graphmert/utils/combine_tails/combine_tails.py`, `prompts/combine_tails.yaml`, `scripts/phases/graphmert.sh` (validate_predictions step)

---

## 1. TL;DR

`combine_tails` currently runs an **LLM yes/no "scientific plausibility filter"** (`is_valid = text.startswith("yes")`, keep only `llm_valid==True`). Per **four independent sources** (upstream code, the neuroscience paper, and the README in two places) its job is to **merge all shard predictions and deduplicate** — *not* filter. Validation is a **separate** step (`fact_score`, the two-LLM consensus gate).

Because the filter's prompt still asks the model to "combine candidate tokens → output a JSON list" while the parser checks for `"yes"`, **every triple is rejected**. This has been true since the initial release (33823cb, 2026-06-10). Net effect: **GraphMERT has contributed 0 triples to the KG on every run** — the curriculum has always been built from the seed KG alone.

## 2. The observed failure (Space smoke, 2026-07-01)

- `predict_tails` produced 36 candidate triples (4 with non-empty tails).
- `combine_tails` emitted **0** rows to `final_kg_scientific_only.csv`.
- `fact_score` therefore validated **0** (`Validated (both): 0 (0.0%)`, "below expected range 15k–50k").
- `expand_kg` merged seed + 0 → final KG = **385 triples = seed only**.
- `curriculum.path_traversal` → all triples hop-0 → `generate_qa_pair` died: *"No paths found for hops 1–3."*

Raw LLM output (captured via a temporary debug, since removed) confirmed the mechanism: on all 4 real triples the model reasoned about *"combining candidate tokens into biological entities"* and emitted a JSON list — never `"yes"` — so `startswith("yes")` was `False` for all. `closed_think=True` ruled out truncation; the parser and token budget were both red herrings.

## 3. What combine_tails is SUPPOSED to do — four unanimous sources

| Source | combine_tails role | validation role |
|---|---|---|
| **graphmert_umls** (`dev`/`main`/`origin/main`/`upstream/main`, byte-identical) | token-combiner: prompt "combine candidate tokens → output `["tail", …]`"; parser `extract_rightmost_list` (`json.loads`/`ast.literal_eval`) | separate — `llm_evaluation_scores/{fact_score,validity_score}.py` |
| **Neuro paper** (2605.25183v2, Stephen & Jha) | no token-combine needed (Qwen3-14B extracts full entities); GraphMERT MNM expands | **two-LLM consensus** (both must vote Yes) — the sole validity gate |
| **Neuro README — repo layout** | `combine_tails/` → *"merge predicted tails"* | `llm_scores/` → *"two-LLM fact validation"* |
| **Neuro README — Part 2, Step 7** (main) | *"**Merge all shard predictions and deduplicate**"* → writes `combined/expanded_triples.csv` | *"Score each candidate triple with two independent LLMs — keep only triples both models agree are factually supported"* → `validated_triples.csv` (expected **15k–50k**) |

**No source describes combine_tails as a plausibility filter.** The yes/no filter is 100% a neuro_SI_pipeline invention.

## 4. Provenance — how the bug got in

- **33823cb (2026-06-10, initial release):** the port kept the upstream combine prompt (`from combine_tokens_prompts import SYSTEM_CONTEXT as SYSTEM_PROMPT`) but **swapped the parser** `extract_rightmost_list` → `startswith("yes")` and re-labeled the function `filter_scientific_triples` / "scientific plausibility filtering". Mismatch from commit 1.
- **3ea615e (2026-06-20):** prompt migrated to `prompts/combine_tails.yaml` "bit-identically" — faithfully preserved the wrong (combine) prompt.
- **`main`:** same mismatch, prompt still hardcoded (not migrated). Broken too.
- Deliberate *intent* was clearly a filter (the renamed function, `llm_valid`, `scientific_only` outputs), but that intent was itself wrong — it duplicates `fact_score` and contradicts the README/paper/upstream. So: **a mistake, not a hidden design.**

## 5. Two concrete defects vs the README

1. **Wrong operation:** filters (`startswith("yes")`) instead of merge+dedup.
2. **Wrong output contract:** README says combine_tails writes `combined/expanded_triples.csv` (all merged candidates); the code writes `combined/final_kg_scientific_only.csv` (a *filtered subset*), and `graphmert.sh` wired `fact_score` to read that filtered file. So `fact_score` receives a pre-filtered set, not all merged candidates.

## 6. The fix (target state)

`combine_tails` should:
1. Load all shard CSVs (already does — `load_all_shard_csvs`).
2. **Merge + deduplicate** the predicted (head, relation, tail) rows.
3. Write **all** merged candidates to `combined/expanded_triples.csv` (README name).
4. **Remove** the `startswith("yes")` / `llm_valid` / `scientific_only` filter path entirely.

`fact_score` remains the sole validity gate (two-LLM consensus), reading `expanded_triples.csv`. `graphmert.sh` fact_score `--input_csv` updated to `expanded_triples.csv`.

## 7. OPEN DECISION — merge semantics (A-lite vs A-upstream)

The README keeps `--model_id qwen3-14b` on combine_tails, implying an LLM is involved in the merge. Two readings:

- **A-lite — plain merge/dedup (no LLM).** Concatenate shards, dedup/normalize (head,relation,tail), write `expanded_triples.csv`. Drop the vLLM load entirely. Simplest; avoids an LLM step immediately before another LLM step (fact_score). *Recommended if predict_tails emits complete tails.*
- **A-upstream — LLM combine (restore upstream).** Restore `extract_rightmost_list` + a combine prompt so the LLM merges candidate **tokens** into full entity names, then dedup. Faithful to graphmert_umls; only meaningful if the tail-producer emits token-level candidates.

**Deciding factor:** does neuro's `predict_tails_llm` emit *complete tails* (→ A-lite) or *token-level candidates that need combining* (→ A-upstream)? The README's Step 6 quality-check says "top-20 predicted **tokens** per head" (token-level), but the observed data (`deep space network`, `solar system`) are complete entities. This ambiguity is the one thing worth confirming.

## 8. Questions for Jake (both well-evidenced now)

1. **Merge semantics:** combine_tails per the README = "merge + deduplicate". Is the `qwen3-14b` LLM there to *combine candidate tokens into entity names* (A-upstream), or is a plain dedup enough (A-lite) now that tails may arrive complete?
2. **Which producer feeds validation (paper alignment):** the paper's Phase 3 expands with **GraphMERT MNM** predictions (= our `predict_tails_gm`) validated by the two-LLM consensus. The code instead validates a separate `predict_tails_llm` (not in the paper) while `predict_tails_gm`'s output is **unmerged** ([graphmert.sh:265](../scripts/phases/graphmert.sh#L265)). Should the graphmert phase follow the paper — GraphMERT MNM → merge → fact_score → expand — and retire `predict_tails_llm` + the filter?

## 9. Blast radius (when the fix is approved)

- `2_graphmert/utils/combine_tails/combine_tails.py` — replace filter with merge+dedup; output `expanded_triples.csv`.
- `prompts/combine_tails.yaml` — retire (A-lite) or rewrite to a combine prompt (A-upstream); fix stale header comment.
- `scripts/phases/graphmert.sh` — `fact_score --input_csv …/combined/expanded_triples.csv`.
- `configs/default.yaml` — the `combine_tails_max_tokens` / `combine_tails_no_think` knobs become dead if A-lite (drop them); keep if A-upstream.
- `step_quality.py` / probes — none reference combine_tails counts (verified).

## 10. Interim state (already done, keeps smoke unblocked)

- **TEMP debug removed** from combine_tails.py (tree clean).
- **`calculate_hops` seed-only fallback** broadened to trigger on all-hop-0 (not just empty) — so smoke's curriculum runs off the seed KG despite 0 expansion. Committed. This is what keeps the pipeline green while combine_tails stays broken-pending-decision.
- **`combine_tails_max_tokens` config knob** (added while chasing the truncation red herring) is harmless and can stay or be reverted with the A-lite path.
- **`predict_tails.py` `id` fix** (KeyError) — unrelated, correct, keep.

**Do NOT** rewrite combine_tails until §7 is decided. It fails loud enough now (0 validated, "below 15k–50k" warning) that it won't silently regress.
