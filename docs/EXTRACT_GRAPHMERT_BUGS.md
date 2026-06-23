# Extract + GraphMERT Bug Ledger

> Living doc. New extract/graphmert bugs land here with date, file:line refs,
> and fix detail. Sectioned chronologically (most recent first) so the
> latest incident is at the top; older bugs accumulate below.

## TL;DR

| Session date | Bugs surfaced | Trigger | Status |
|---|---|---|---|
| **2026-06-22 (smoke run)** | **14** | First end-to-end `extract → graphmert` run on this fork (`neuro_SI_pipeline` HEAD) | All fixed in source; one apt dep needs bootstrap.sh patch |
| 2026-06-22 (pre-smoke session) | 4 | Same-day prep work — Cython + venv deps + Fix A.2 grounding + entity_discovery domain port | All fixed |
| 2026-05-27 to 2026-05-30 (Stage 1 medical pipeline) | 5 documented | Recipe B working spec / first RunPod runs | All fixed; pinned in memory |

**Total catalogued: 23 extract/graphmert bugs across 3 incident windows.**

Recurring pattern (true across all sessions): producer↔consumer contracts
in `scripts/phases/graphmert.sh` and Python consumers diverge silently
because the chain isn't run end-to-end after merges. Filenames, column
names, file formats, model IDs, and config flags all need alignment.

### Counts by category (2026-06-22 smoke run only)

| Category | Count | Notes |
|---|---|---|
| Environment / system | 1 | apt package missing on pod |
| Configuration | 5 | `args_mlm.yaml` + `configs/default.yaml::models` |
| Code defensive | 3 | None-guards + Python scoping shadow removals |
| Producer ↔ consumer wiring | 4 | filename + column + file-format mismatches |
| LLM parser / sampling | 1 | Qwen3 `<think>` block parsing + token budget |
| **Subtotal** | **14** | |

### Index: every catalogued bug (chronological)

| # | Date | Code | Phase / file | One-line |
|---|---|---|---|---|
| 23 | 2026-06-22 | E1 | preprocess / pyximport | `Python.h` missing — apt `python3.10-dev` |
| 22 | 2026-06-22 | C1 | train_mnm / args_mlm.yaml | `mlm_sbo=true` with TailSlot collator (no pairs) |
| 21 | 2026-06-22 | C2 | train_mnm / args_mlm.yaml | `metric_for_best_model=eval_loss` crashes when eval N=2 |
| 20 | 2026-06-22 | C3 | predict_tails / graphmert.sh | `MLM_CHECKPOINT` default points GraphMERT path → vLLM can't load |
| 19 | 2026-06-22 | C4 | validate / configs.models | `openai/gpt-oss-20b` MXFP4 incompatible with vLLM 0.7.3 |
| 18 | 2026-06-22 | C5 | validate / configs.models | `mistralai/Mistral-Nemo-12B` is not a valid HF repo (404) |
| 17 | 2026-06-22 | D1 | model / modeling_graphmert.py | Missing None-guard for `pairs` in `lm_pair_head` call |
| 16 | 2026-06-22 | D2 | predict_tails_llm.py | Function-local `from pipeline_config import` shadows module-level |
| 15 | 2026-06-22 | D3 | combine_tails.py | Same scoping shadow — already firing as UnboundLocalError |
| 14 | 2026-06-22 | W1 | combine_tails.py | Filename filter `"exploded"` doesn't match actual `predictions_shard*` |
| 13 | 2026-06-22 | W2 | graphmert.sh | `expanded_triples.csv` passed to fact_score, no producer writes it |
| 12 | 2026-06-22 | W3 | merge_kgs.py | Hardcoded `read_parquet` can't ingest `validated_triples.csv` |
| 11 | 2026-06-22 | W4 | merge_kgs.py | Required `source`/`target` cols; graphmert pipeline uses `head`/`tail` |
| 10 | 2026-06-22 | L1 | combine_tails.py | Qwen3 `<think>` block + `max_tokens=10` → 100% rejection |
| 9 | 2026-06-22 (early) | H1 | 2_graphmert/requirements.txt | `Cython==3.2.4` missing for pyximport |
| 8 | 2026-06-22 (early) | H2 | 3_si_curriculum/requirements.txt | `networkx==3.4.2`, `vllm==0.7.3` missing |
| 7 | 2026-06-22 (early) | H3 | dataset_preprocessing_utils.py | Fix A.2 `_norm_head` at inner sites (0 → 104+2 grounded) |
| 6 | 2026-06-22 (early) | H4 | prompts/ + entity_discovery.py | `entity_discovery` running diabetes prompts on neuroscience text |
| 5 | 2026-05-30 | H5 | graphmert_smoke.yaml | `learning_rates: [list]` SLURM-only; HF default 5e-5 silently wins off-SLURM |
| 4 | ~2026-05-28 | H6 | upstream / dataset_preprocessing | `cut_dataset_for_testing` flag asymmetric + buggy in `tokenization_utils.py:291` |
| 3 | ~2026-05-28 | H7 | jha-lab/filtered_UMLS load | `mrrel.csv` has int > 2^53 → pyarrow CSV inference fails |
| 2 | ~2026-05-28 | H8 | runpod base image | CUDA driver 12.4 vs torch cu128 mismatch → silent CPU fallback |
| 1 | 2026-05-27 | H9 | upstream RUNPOD_REPRODUCTION_GUIDE | No `setup.py`; Cython compiles at runtime via pyximport |

---

## 1. Environment

### E1 — `Python.h` missing on RunPod image (Cython compile blocker)

- **Symptom**: `run_dataset_preprocessing.py` failed at module import with
  `fatal error: Python.h: No such file or directory` deep inside
  `pyximport`'s build of `algos_graphmert.pyx`.
- **Root cause**: pyximport runtime-compiles `.pyx` → `.c` → `.so`; that
  requires `python3.10-dev` (Debian/Ubuntu apt package providing the
  Python C headers). The pip-installed `Cython==3.2.4` (added to
  `2_graphmert/requirements.txt` earlier today) provides the Cython
  compiler but NOT the Python.h header.
- **Fix**: `apt install -y python3.10-dev` on the live pod (no source
  change; future fix: add to `scripts/runpod/bootstrap.sh` apt list +
  add a Python.h preflight gate).
- **Hours wasted**: ~40 min preprocess re-run risk if not caught.

---

## 2. Configuration

### C1 — `mlm_sbo=true` with `GraphMertTailSlotDataCollator`

- **Symptom**: `train_mnm` crashed at step 0 with
  `AttributeError: 'NoneType' object has no attribute 'size'` inside
  `modeling_graphmert.py:927` (`pairs.size()`).
- **Root cause**: this fork's `mlm_utils.py:239` wires a custom
  `GraphMertTailSlotDataCollator` (`mlm_utils.py:102`) that implements a
  tail-slot training objective and emits batches with `input_nodes /
  attention_mask / leaf_relationships / head_lengths / labels` — **no
  `pairs` field**. But the upstream `args_mlm.yaml` `config_overrides`
  string set `mlm_sbo=true` and the top-level `mlm_sbo: true` likewise.
  Result: `config.mlm_sbo=True → model.use_sbo=True`, so
  `modeling_graphmert.py:1203-1204` unconditionally called
  `self.lm_pair_head(outputs, pairs)` with `pairs=None`.
- **Where the gap was created — provenance of the model code**: this
  fork's `mlm_utils.py` and its custom `GraphMertTailSlotDataCollator`
  have been here since `33823cb` (the initial public release). The
  matching `2_graphmert/graphmert_model/` directory — `configuration_graphmert.py`,
  `modeling_graphmert.py`, `collating_graphmert.py`, `algos_graphmert.pyx` —
  was added MUCH later in commit `2d7b782` ("debug ID type error and
  graphmert model inclusion"). That merge dropped the model sources into
  this fork **without reconciling them against the pre-existing
  `mlm_utils.py` design choices**:
    - `mlm_utils.py` chose TailSlot collator over the upstream Cython
      SBO collator (deliberate fork divergence).
    - But the freshly-merged `modeling_graphmert.py` was the upstream
      `graphmert_umls` model code AS-IS, which assumes SBO pairs are
      always provided when `mlm_sbo=True`.
    - The merge also didn't update `args_mlm.yaml` (still inherited
      `mlm_sbo=true` from upstream).
  Net effect: the SBO branch in the model expected data that this
  fork's collator never produces. Bug stayed latent because train_mnm
  hadn't been run end-to-end since `2d7b782` landed — surfaced today
  the moment the chain ran.
- **Fix**: [args_mlm.yaml:27](2_graphmert/launch_configs/args_mlm.yaml#L27)
  and [args_mlm.yaml:69](2_graphmert/launch_configs/args_mlm.yaml#L69) both
  flipped to `mlm_sbo=false`. Comment block explains the fork-vs-upstream
  divergence (graphmert_umls upstream has `mlm_sbo=true` with its full
  Cython collator; we use TailSlot instead).
- **Defense in depth**: paired with **D1** below — None-guard added at
  `modeling_graphmert.py:1209` so a future merge that re-introduces
  `mlm_sbo=true` without reconciling the collator will degrade
  gracefully instead of crashing.

### C2 — `metric_for_best_model: eval_loss` crashes when eval N=2

- **Symptom**: `train_mnm` got to step 25 then crashed with
  `KeyError: 'eval_loss'` from
  `transformers.trainer._determine_best_metric`.
- **Root cause**: at smoke scale, eval has only 2 rows. The TailSlot
  collator's `_pick_head_pos` randomly picks a head with a non-pad leaf
  slot to mask — and on 2 rows often returns `None` for both → all labels
  remain `-100` → CrossEntropyLoss output isn't recorded as `eval_loss`
  in HF Trainer's metrics dict. HF Trainer's `_determine_best_metric` is
  called on **every** eval whenever `metric_for_best_model is not None`,
  regardless of `load_best_model_at_end`.
- **Initial misdiagnosis**: setting `load_best_model_at_end: false` is
  NOT sufficient — the gating condition is `metric_for_best_model is not
  None`, not the load flag. Both have to be cleared.
- **Fix**: [args_mlm.yaml:131-133](2_graphmert/launch_configs/args_mlm.yaml#L131-L133)
  set `load_best_model_at_end: false` AND `metric_for_best_model: null`.
  Comment block documents the gate so re-enabling on pilot/paper is one
  line. `metric_for_best_model: eval_loss` (kept declaratively in
  comments) is recoverable when eval splits are large enough that
  `_pick_head_pos` reliably succeeds.

### C3 — `predict_tails` wired to GraphMERT checkpoint (vLLM expects Qwen)

- **Symptom**: After train_mnm produced
  `outputs/graphmert/checkpoints/best/`, `predict_tails_llm.py` tried
  `vllm.LLM(model="<graphmert checkpoint path>")` and would crash with
  `ValueError: The checkpoint you are trying to load has model type
  'graphmert' but Transformers does not recognize this architecture.`
  (BERT-style MLM via custom `model_type='graphmert'` is not registered
  in vLLM's process; vLLM uses HF `AutoConfig.from_pretrained()` for the
  config-load step, and that step fails before vLLM's own architecture
  registry is even consulted.)
- **Root cause attribution (UPSTREAM, not fork)**: this bug exists on
  Princeton's `main` branch in the same form. Their `main` has all three
  components in mutual inconsistency:
    1. `2_graphmert/predict_tails_llm.py` — uses `vllm.LLM(...)`, docstring
       example shows `--model_id /path/to/qwen3-32b` (intent: LLM predictor)
    2. `2_graphmert/utils/predict_tails.py` — uses
       `GraphMertForMaskedLM.from_pretrained(...)` (intent: GraphMERT predictor)
    3. `README.md` step 2.6 line 365 — `python 2_graphmert/predict_tails_llm.py
       --model_id $OUTPUT_BASE/graphmert/checkpoints/best` (mixes the two:
       calls the LLM-designed script but passes the GraphMERT path)
  If anyone runs README:365 literally on Princeton's main, it hits the
  same `ValueError`. The fork's `scripts/phases/graphmert.sh:166`
  faithfully calls the same script the upstream README documents — so
  the fork-side contribution is only to perpetuate the broken
  invocation, not to introduce it.
- **Empirical proof (reproducible in 30 seconds)**:
  ```python
  # Save config.json declaring model_type=graphmert + architectures=[GraphMertForMaskedLM]
  # to a temp dir, then:
  from vllm import LLM
  llm = LLM(model=tmp, enforce_eager=True, gpu_memory_utilization=0.05, max_model_len=128)
  # Raises:
  # ValueError: The checkpoint you are trying to load has model type `graphmert`
  # but Transformers does not recognize this architecture.
  ```
  The `model_type='graphmert'` IS registered in HF via
  `AutoConfig.register("graphmert", GraphMertConfig)` at
  [mlm_utils.py:46-63](2_graphmert/utils/mlm_utils.py#L46-L63), but
  that registration only fires when `mlm_utils.py` is imported. vLLM
  doesn't import it, so the registration never reaches vLLM's process.
- **Fix**: Added [configs/default.yaml:32](configs/default.yaml#L32)
  `predict_tails: Qwen/Qwen3-14B` (matches `extract` so HF cache is
  reused). [graphmert.sh:48](scripts/phases/graphmert.sh#L48) now
  sources via `PREDICT_TAILS_MODEL_ID=$(get_model_id predict_tails "")`
  (same pattern as `EXTRACT_MODEL_ID` / `VALIDATE_A` / `VALIDATE_B`).
  `MLM_CHECKPOINT` env-var override kept for ad-hoc runs. The fix makes
  the LLM path the active one, NOT the GraphMERT path.
- **Architectural note (superseded 2026-06-23)**: An earlier draft
  here read "the trained GraphMERT checkpoint is never consumed
  downstream — dead disk weight." That framing was wrong. Per Jake
  Stephen's clarification (email 2026-06-22 evening) and his subsequent
  README+code fixes on `main` (commits [1834992](https://github.com/scient-lab/neuro_SI_pipeline/commit/1834992)
  + [41d7c8b](https://github.com/scient-lab/neuro_SI_pipeline/commit/41d7c8b)),
  the design is **two separate predictors that both feed combine_tails**:
    - Step 2.6 `predict_tails_llm.py` — LLM-based (vLLM + Qwen3-32B),
      writes to `predictions/`
    - Step 2.6b `utils/predict_tails.py` — GraphMERT MLM-based, uses the
      trained checkpoint, writes to `predictions_graphmert/`
  The pipeline wasn't missing the LLM path; it was missing step 2.6b
  entirely. Wired in as `step_predict_tails_gm` in
  [scripts/phases/graphmert.sh](scripts/phases/graphmert.sh) on
  2026-06-23. Step skips silently when no trained checkpoint exists
  (smoke runs) and gates on `GRAPHMERT_PREDICT_TAILS_GM_REQUIRED=1`
  for hard-require.
- **Re-classified**: this entry was tagged Case D ("fork-introduced
  modification") in an earlier draft of `BUG_FAILURE_CASES_2026-06-22.md`.
  After the empirical vLLM repro above, the correct classification is
  **Case B** ("code bug that crashes anywhere") — the README+script
  inconsistency exists on Princeton's main, and the fork only
  perpetuates the broken invocation in its shell wrapper. The 2026-06-23
  step-2.6b wiring closes the gap; the missing-step fragment is no
  longer a bug, only an extension.

### C4 — `openai/gpt-oss-20b` MXFP4 quantization incompatible with vLLM 0.7.3

- **Symptom**: `fact_score.py` crashed at vLLM model load with
  `ValueError: Unknown quantization method: mxfp4. Must be one of [...]`.
- **Root cause**: `openai/gpt-oss-20b` uses MXFP4 (microscaling FP4)
  quantization, added to vLLM in 0.10+. This fork pins `vllm==0.7.3`
  (per `[Stage 3 RL 12GB gotchas]` memory) across
  `2_graphmert/requirements.txt` and `3_si_curriculum/requirements.txt`.
  The model id appeared in TWO places:
  `configs/default.yaml::models.validate_a` (graphmert phase) and
  `models.curriculum_check_a` (curriculum phase — would have hit the
  same crash later).
- **Fix**: Both swapped to `Qwen/Qwen3-14B` at
  [configs/default.yaml:26](configs/default.yaml#L26) and
  [configs/default.yaml:48](configs/default.yaml#L48). Comment block notes
  the gpt-oss-20b original; revert when vLLM is upgraded. Two-LLM
  diversity preserved by the b-side (Mistral) staying different.

### C5 — `mistralai/Mistral-Nemo-12B` is not a valid HF repo (404)

- **Symptom**: `fact_score.py` loaded validate_a (Qwen3-14B) successfully,
  computed scores_1, then crashed loading validate_b with
  `huggingface_hub.errors.RepositoryNotFoundError: 404 Client Error.
  Repository Not Found for url: .../mistralai/Mistral-Nemo-12B/...`.
- **Root cause**: `Mistral-Nemo-12B` is marketing shorthand, not a real
  HuggingFace repo id. Mistral's actual repo names are
  `mistralai/Mistral-Nemo-Base-2407` (base) and
  `mistralai/Mistral-Nemo-Instruct-2407` (instruct). The wrong id
  appeared in TWO config slots: `validate_b` and `curriculum_check_b`.
- **Fix**: Both swapped to `mistralai/Mistral-Nemo-Instruct-2407` at
  [configs/default.yaml:27](configs/default.yaml#L27) and
  [configs/default.yaml:49](configs/default.yaml#L49). Instruct variant
  chosen because both consumers (`fact_score.py`, `curriculum_check`)
  prompt with system+user messages.

---

## 3. Code defensive

### D1 — `modeling_graphmert.py:1204` missing None-guard for `pairs`

- **Pattern**: parallel to the existing `pair_labels is not None` guard
  at [modeling_graphmert.py:1206](2_graphmert/graphmert_model/modeling_graphmert.py#L1206).
  The SBO branch at line 1203-1204 unconditionally called
  `self.lm_pair_head(outputs, pairs)` even when `pairs` could legitimately
  be `None` (e.g., from a collator that doesn't emit SBO data).
- **Fix**: Added `and pairs is not None` to the guard at
  [modeling_graphmert.py:1209](2_graphmert/graphmert_model/modeling_graphmert.py#L1209).
  Defense-in-depth alongside the `mlm_sbo=false` config fix (C1) — if
  `use_sbo` is False the branch is skipped anyway, but the guard
  protects against future collators that emit pairs sometimes and not
  others.

### D2 — `predict_tails_llm.py:252` function-local import shadowing module-level

- **Pattern**: classic Python scoping trap. `predict_tails_llm.py:48`
  imports `get_phase_param` at module level. Inside `main()`, line 252
  had `from pipeline_config import get_phase_param` — Python's name
  resolution treats this as a **function-local binding** for the entire
  `main()` body, shadowing the module-level import. Any reference to
  `get_phase_param` in `main()` ABOVE line 252 would have raised
  `UnboundLocalError: local variable 'get_phase_param' referenced before
  assignment`.
- **Lucky escape**: in this file, all earlier usages happen at module
  level (line 48 import + module-scope config reads). `main()` only
  uses `get_phase_param` AT the line 252 site itself, so the shadow was
  never triggered — but the bug was waiting for the first developer to
  add a `get_phase_param` call elsewhere in `main()`.
- **Fix**: Removed the redundant in-function import; rely on the
  module-level import at line 48. Comment added warning future
  contributors not to re-add the local import.

### D3 — `combine_tails.py:200` function-local import shadowing module-level

- **Pattern**: same as D2. Module-level import at line 37 was shadowed
  by a function-local re-import inside `main()`. Unlike D2, this one
  **DID fire** in production today: `get_phase_param` was called at
  line 189 (above the line 207 re-import), raising `UnboundLocalError`.
- **Fix**: Removed the in-function import (lines 207 surroundings,
  including the dead `sys.path` defensive setup). Same warning comment
  as D2.

---

## 4. Producer ↔ consumer wiring

This is the most common pattern from today's run — 4 distinct mismatches
where the producer wrote one filename / format / column-naming and the
consumer expected another.

### W1 — `combine_tails.py` filename filter

- **Producer**: `predict_tails_llm.py:223` writes
  `predictions_shard{N}_of{M}.csv`.
- **Consumer (was)**: `combine_tails.py:96` filtered for
  `"exploded" in f` — matched nothing. (The variable name `out_exploded`
  in the producer hints that the convention changed at some point in
  upstream history; the consumer wasn't updated.)
- **Fix**: [combine_tails.py:104](2_graphmert/utils/combine_tails/combine_tails.py#L104)
  now filters for `"predictions_shard" in f`. `FileNotFoundError`
  message also rewritten to explicitly name the producer file pattern so
  future drift surfaces fast.

### W2 — `graphmert.sh` fact_score input filename

- **Producer**: `combine_tails.py:214` writes
  `final_kg_scientific_only.csv` (post-LLM scientific filter,
  head/relation/tail columns).
- **Consumer (was)**: `graphmert.sh:191` passed
  `--input_csv "$GRAPHMERT_DIR/combined/expanded_triples.csv"` to
  `fact_score.py`. **No producer in the repo writes `expanded_triples.csv`.**
- **Fix**: [graphmert.sh:211](scripts/phases/graphmert.sh#L211) now passes
  `final_kg_scientific_only.csv`. `fact_score.py:10` docstring example
  already showed this as the expected input — wiring just hadn't caught
  up.

### W3 — `merge_kgs.py` format assumption (parquet only)

- **Producer**: `fact_score.py:158` writes CSV via
  `validated.to_csv(args.output_csv, index=False)`.
- **Consumer (was)**: `merge_kgs.py:105` hardcoded
  `return pd.read_parquet(path)`. Crashed on
  `pyarrow.lib.ArrowInvalid: Parquet magic bytes not found in footer.`
- **Fix**: [merge_kgs.py:108](1_seed_kg/merge_kgs.py#L108) now sniffs by
  file extension: `pd.read_csv(path) if path.suffix == ".csv" else
  pd.read_parquet(path)`. Function name kept as `_load_parquet` for
  git-history clarity even though it now handles both. Error message
  generalized.

### W4 — `merge_kgs.py` column-name assumption (source/target only)

- **Producer**: graphmert pipeline (predict_tails → combine_tails →
  fact_score) carries the KG triple convention `head` / `relation` /
  `tail` throughout.
- **Consumer (was)**: `merge_kgs.py:120` required `source` / `target`
  columns (graphrag's edge-view convention). Crashed with
  `KeyError: "[new] missing 'source'; got ['id', 'head', 'relation',
  'tail', 'query_had_no_tails']"`.
- **Fix**: [merge_kgs.py:120-130](1_seed_kg/merge_kgs.py#L120-L130) now
  accepts either column-pair convention and normalizes `head`/`tail` →
  `source`/`target` for the rest of the module.

---

## 5. LLM parser / sampling

### L1 — `combine_tails.py` 100% rejection due to Qwen3 think-block parsing

- **Symptom**: combine_tails ran successfully, processed 13 input triples
  (textbook-valid neuroscience like "calyx of held located_in brainstem"
  and "synaptic vesicles originates_from nerve terminals"), but flagged
  ALL 13 as `llm_valid=False`. `final_kg_scientific_only.csv` was 41B
  (just a header).
- **Two compounding bugs**:
  1. **Token budget**: `max_tokens=10` cut Qwen3 off mid-`<think>` block.
     With thinking on (default — docstring states "disabling thinking
     destroys quality"), Qwen3 emits `<think>...reasoning...</think>YES`
     and 10 tokens isn't enough to reach the answer.
  2. **Parser**: `text.startswith("yes") or text.startswith("true")`
     checked the **raw** response. Even with enough token budget, the raw
     response starts with `<think>` — so the parser returned False on
     every triple regardless of content.
- **Fix**: [combine_tails.py:142](2_graphmert/utils/combine_tails/combine_tails.py#L142)
  bumped `max_tokens=10 → 512`. [combine_tails.py:152-157](2_graphmert/utils/combine_tails/combine_tails.py#L152-L157)
  strip `<think>...</think>` with `re.sub(..., flags=re.DOTALL)` before
  the `.startswith("yes")` check. Verified on 7 synthetic cases including
  Qwen3-shaped thinking outputs.

---

## 6. Behavioral concerns flagged (not fixed today)

### B1 — `predict_tails` declines many queries (`query_had_no_tails=True` for 23/36)

Smoke-scale: predict_tails fired 36 (head, relation) queries to Qwen3
and got "no plausible tail" on 23 of them — only 13 produced actual
triples. Whether this is a Qwen3-prompt issue or genuine "no good tail
in the seed corpus" is unclear without running pilot scale. Defer to
post-pilot evaluation.

### B2 — `train_mnm` produces orphaned checkpoint

The active downstream prediction path is LLM-based (vLLM + Qwen3,
script `predict_tails_llm.py`). The trained GraphMERT MLM checkpoint at
`outputs/graphmert/checkpoints/best/` is **never read** by any
downstream consumer in this fork. Either (a) `train_mnm` should be
skipped on smoke/pilot until a downstream consumer is wired (saving
~5 min smoke / hours pilot), or (b) `step_predict_tails` should switch
to `utils/predict_tails.py` (the dormant GraphMERT-based variant) for a
proper KG-trained tail predictor — needs Princeton-lab alignment call.

### B3 — `analysis.sh --phase graphmert` is a stub

The shell script explicitly says "STUB — full quality analysis pending"
and just lists artifacts. Quality metrics for the graphmert phase
(coverage, precision proxies, eval-loss curves, KG size growth) aren't
implemented yet. Tracked as a separate workstream.

---

## 7. Files changed

| File | Change |
|---|---|
| `2_graphmert/launch_configs/args_mlm.yaml` | mlm_sbo=true→false (×2); load_best_model_at_end=true→false; metric_for_best_model=eval_loss→null |
| `2_graphmert/graphmert_model/modeling_graphmert.py` | Added `pairs is not None` guard at line 1209 |
| `2_graphmert/predict_tails_llm.py` | Removed function-local `from pipeline_config import get_phase_param` (~7 lines net) |
| `2_graphmert/utils/combine_tails/combine_tails.py` | (a) filename filter `exploded`→`predictions_shard` (b) removed function-local import shadow (c) max_tokens=10→512 + strip Qwen3 think block |
| `1_seed_kg/merge_kgs.py` | (a) `_load_parquet` sniffs CSV vs parquet by extension (b) `_ensure_cols` accepts `head`/`tail` alongside `source`/`target` |
| `scripts/phases/graphmert.sh` | (a) added `PREDICT_TAILS_MODEL_ID` sourced from `get_model_id('predict_tails')` (b) `step_predict_tails` uses the new var (c) `expanded_triples.csv` → `final_kg_scientific_only.csv` for fact_score input |
| `configs/default.yaml` | (a) added `models.predict_tails: Qwen/Qwen3-14B` (b) `validate_a`: gpt-oss-20b → Qwen3-14B (c) `curriculum_check_a`: gpt-oss-20b → Qwen3-14B (d) `validate_b`: Mistral-Nemo-12B → Mistral-Nemo-Instruct-2407 (e) `curriculum_check_b`: same |
| `scripts/runpod/bootstrap.sh` | (not yet) — should add `python3.10-dev` to apt list + Python.h preflight |

---

## 8. Lessons

1. **"First end-to-end run reveals everything"** — six of these bugs are
   the same shape: producer/consumer naming/format/column/model-id
   contracts that look obviously broken once you see them but stayed
   silent for months because no one had connected the dots.

   **Where these contracts were broken: the upstream merge gap.**
   Commit `2d7b782` ("debug ID type error and graphmert model inclusion")
   imported `2_graphmert/graphmert_model/*` (the Cython collator, the
   modeling code, the configuration) directly into this fork without
   reconciling against the pre-existing fork-local design choices that
   `mlm_utils.py`, `combine_tails.py`, `predict_tails_llm.py`,
   `merge_kgs.py`, and `graphmert.sh` had already committed to since
   `33823cb`. Concretely, the merge:
     - Did not retag `args_mlm.yaml` (still has the upstream
       `mlm_sbo=true` even though the fork's `mlm_utils.py` uses a
       collator that never produces SBO data) → **C1**.
     - Did not update `predict_tails_llm.py`'s `--model_id` wiring even
       though the upstream's GraphMERT-based predict_tails was replaced
       by this fork's LLM-based variant → **C3**.
     - Did not align column-naming conventions: upstream `merge_kgs.py`
       uses `source`/`target`; this fork's downstream graphmert chain
       uses `head`/`tail` → **W4**.
     - Did not align output-format conventions: upstream `merge_kgs.py`
       expects parquet inputs; this fork's `fact_score.py` writes CSV
       → **W3**.
     - Did not align filename conventions: `combine_tails.py`'s filter
       looks for `"exploded"`; the fork's `predict_tails_llm.py` writes
       `predictions_shard*.csv` → **W1**.
     - Did not align argument conventions: `graphmert.sh` passes
       `expanded_triples.csv` to `fact_score.py` while
       `combine_tails.py` writes `final_kg_scientific_only.csv` → **W2**.

   In short: the model code was dropped in without the matching context
   (config alignment, downstream wiring updates, naming-convention sweep)
   that would have made it compatible with the existing fork. Process
   gap, not a code quality issue per se — but it produced the 6 wiring
   bugs that interrupted today's smoke run.

2. **Function-local imports are scoping landmines** — Python's
   "any assignment in a function makes the name local for the entire
   function" rule bit us twice (D2, D3). Convention to adopt: never
   re-import a name inside a function if it's already at module level.
   Linter rule worth considering.
3. **vLLM's pinned 0.7.3 is a constraint, not a default** — half of
   today's config bugs (C4, C5, partially C3) stemmed from model IDs
   being chosen without checking vLLM compatibility. Worth a `make
   verify-model-ids` task that does a dry-run model load for each entry
   in `configs/default.yaml::models`.
4. **`stats.sh` and `pipeline.sh` together created a manifest/disk-state
   ambiguity** — when we manually ran `run_dataset_preprocessing.py`
   outside `pipeline.sh` to recover from E1, the manifest still showed
   `preprocess = failed`. `manifest.py end-step --exit-code 0`
   reconciled it. The cleaner architectural fix: an idempotency wrapper
   that checks "does the output already exist? mark completed" so manual
   recovery doesn't desync the source-of-truth.
5. **Audit downstream proactively, not reactively** — once we'd hit 6
   bugs reactively, the user pushed back and asked for a proactive
   audit. That audit (run as a background agent) found 10 ACTIVE
   crashers and 3 latent issues in the SFT/RL phases ahead. Doc at
   [docs/AUDIT_SFT_RL_HANDOFFS_2026-06-22.md](AUDIT_SFT_RL_HANDOFFS_2026-06-22.md).

6. **Merging upstream code is not just `cp -r` — it's a reconciliation
   task.** The 2d7b782 merge brought in `graphmert_model/*` as a code
   drop but the wiring, config, and conventions were never reconciled.
   Process improvements to consider before the next upstream merge:
   - A pre-merge **contract diff**: list every cross-module producer/
     consumer pair touched by the incoming code (filenames, column
     names, model IDs, file formats, config keys) and verify they
     match. Today's 6 wiring bugs would have caught at this stage.
   - An end-to-end **smoke-validation gate**: no merge of model/training
     code lands without a full `extract→graphmert→curriculum` smoke
     run on the fork's actual collator/data path. The bugs we hit
     today were latent for weeks because no one ran the chain.
   - **Provenance comments at merge sites**: every file dropped in via
     `cp -r` from `graphmert_umls` should carry a header comment
     identifying the upstream commit SHA + path so future drift is
     diagnosable. Currently the merged files in `2_graphmert/graphmert_model/`
     have no upstream provenance metadata.

---

## 9. Earlier 2026-06-22 — same-day pre-smoke fixes

These landed before the smoke run started; they're documented elsewhere
but indexed here for the ledger.

### H1 — `Cython==3.2.4` missing from `2_graphmert/requirements.txt`

- **Symptom**: `from . import algos_graphmert` (collating_graphmert.py:27)
  failed because pyximport couldn't find Cython.
- **Root cause**: upstream `graphmert_umls/requirements.txt` had Cython
  pinned; this fork's `2_graphmert/requirements.txt` didn't. Latent until
  the first pyximport invocation actually ran on a fresh venv.
- **Fix**: Pinned `Cython==3.2.4` (matches upstream commit).
- **Related**: precursor to today's **E1** (`Python.h` apt dep); fixing
  the pip side wasn't enough — apt-level dev headers still missing.

### H2 — `networkx==3.4.2` + `vllm==0.7.3` missing from `3_si_curriculum/requirements.txt`

- **Symptom**: `calculate_hops.py` and `generate_questions.py` failed at
  import time on freshly-bootstrapped pods.
- **Root cause**: Both deps arrived transitively via other packages
  before, but the resolver picked different versions on fresh pods.
- **Fix**: Pinned exact versions matching upstream
  `bottom-up-superintelligence/env_setup.sh`.

### H3 — `_norm_head` not applied at all co-occurrence lookup sites

- **Symptom**: `run_dataset_preprocessing.py` reported `Grounding
  results: {'success': 0, ...}` — every triple in the seed KG was
  marked as "no_head_match" against the train text. Step 4 succeeded
  but produced an empty dataset; train_mnm crashed on empty batches.
- **Root cause**: The outer head-match path applied `_norm_head`
  (strip trailing `,.;:!?- ` + lowercase) but the inner co-occurrence
  loop didn't. The seed KG had heads with curly punctuation
  ("astrocytes,") that the inner lookup couldn't match.
- **Fix**: `dataset_preprocessing_utils.py:138` + `:148-149` apply
  `_norm_head(s)` consistently. After: 104 train + 2 eval grounded
  examples on smoke corpus.

### H4 — `entity_discovery` running diabetes prompts on neuroscience text

- **Symptom**: graphmert.preprocess produced obviously-wrong entity
  lists (insulin, glucose, pancreas, etc.) for a neuroscience textbook
  corpus.
- **Root cause**: `2_graphmert/utils/entity_discovery/entity_discovery_prompts.py::SYSTEM_CONTEXT`
  was hardcoded with diabetes domain content (left over from an earlier
  medical-pipeline experiment). The fork's domain was neuroscience but
  this prompt was never updated.
- **Fix**: Initial smoke-blocker that motivated the full prompt YAML
  migration. Now ported to `prompts/entity_discovery.yaml` with
  `{{slot}}` substitution from `domains/neuroscience.yaml`. See
  [PROMPT_MIGRATION.md](PROMPT_MIGRATION.md) §1 + §2 for the full
  inventory.

---

## 10. Historical (pre-2026-06-22)

Less detail per entry — these are pre-existing memory rules with
working fixes. Documented here to make the ledger comprehensive.

### H5 — `learning_rates: [list]` SLURM-only, off-SLURM silently uses 5e-5

- **Discovered**: 2026-05-29 during Recipe B medical pipeline run.
- **File**: `graphmert_umls/utils/mlm_utils.py:90-91`. Off-SLURM the
  list form is ignored; HF `TrainingArguments`' default `learning_rate
  = 5e-5` wins. Visible only by checking output dir name
  `bs<X>_lr_5e-05` instead of intended `bs<X>_lr_0.0004`.
- **Fix**: Use scalar `learning_rate: !!float 0.0004` in YAML; keep
  `learning_rates: [...]` only as SLURM fallback.
- **Memory rule**: [[mlm-lr-scaling-bug]] — full detail.

### H6 — `cut_dataset_for_testing` flag is asymmetric and buggy in upstream

- **Discovered**: 2026-05-27 first RunPod run.
- **Files**: `dataset_preprocessing_utils.py:270-272` (safe — uses
  `min(1000, len(...))`); `tokenization_utils.py:291-292` (unbounded —
  `select(range(1000))` → IndexError if smoke-subset has <1000 rows).
  Gemini/UMLS/entity-discovery steps don't honor the flag at all.
- **Fix**: Set `cut_dataset_for_testing: false` and subset at corpus
  download time instead.
- **Memory rule**: [[graphmert-runpod-gotchas]] — bullet 3.

### H7 — `jha-lab/filtered_UMLS` cannot be loaded via `datasets.load_dataset`

- **Discovered**: 2026-05-28.
- **Root cause**: `mrrel.csv` has integers > 2^53 (e.g.
  `16058431000119104`); pyarrow's CSV-schema inference picks too small
  a type and aborts with `ArrowInvalid: Integer value not in range`.
- **Fix**: `huggingface_hub.snapshot_download(repo_id=...,
  repo_type="dataset", allow_patterns=["*.csv"])` and symlink to
  `$DATA/umls_data/`. Downstream reads with pandas (no overflow).
- **Memory rule**: [[graphmert-runpod-gotchas]] — middle bullet.

### H8 — RunPod base image CUDA 12.4 vs torch cu128 mismatch

- **Discovered**: 2026-05-28.
- **Symptom**: `torch.cuda.is_available()` returns `False`, training
  silently falls back to CPU. Driver API 12.4 (`found version 12040`)
  too old for torch built against cu128.
- **Fix**: Use cu124 base image (`runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`)
  or force-reinstall torch from cu124 wheels.
- **Memory rule**: [[graphmert-runpod-gotchas]] — CUDA bullet.

### H9 — Upstream `RUNPOD_REPRODUCTION_GUIDE.md` claims `python setup.py
build_ext --inplace` works

- **Discovered**: 2026-05-27 first first-time read of the upstream
  guide.
- **Root cause**: No `setup.py` exists in the graphmert_umls repo.
  Cython compiles at runtime via pyximport (see today's **E1** for
  the apt-side gotcha).
- **Fix**: Skip the setup.py step. Pre-trigger with `python -c "from
  graphmert_model import collating_graphmert"`.
- **Memory rule**: [[graphmert-runpod-gotchas]] — first bullet.

### H10 — `profile-key-name-trap` (related: half-day medical-pipeline OOM)

- **Discovered**: ~2 weeks before 2026-06-22 on medical pipeline.
- **Root cause**: Profile YAML keys (`n_completions`, `max_completion_tokens`)
  didn't match what `rl_training.py` actually reads
  (`num_generations`, `max_completion_length`). Merged config faithfully
  carried the wrong keys, code used defaults → 12 GB RTX 3060 OOM.
- **Fix**: Audit profile YAML against `get_phase_param(...)` call sites
  in consumer code. Confirm with merged-config dump.
- **Memory rule**: [[profile-key-name-trap]] — full process.
- **Why it's here**: same shape as today's **C1/C3** (config knob
  doesn't match runtime code). Mentioned to show the pattern predates
  the upstream-merge gap and is recurring.

### H11 — `venv-isolation-rule` (related: medical-pipeline torch/cu mismatch)

- **Discovered**: medical pipeline biomed branch, multi-hour debug.
- **Root cause**: Adding ML-heavy packages (marker-pdf, etc.) to an
  existing training venv (graphrag/graphmert/si_curriculum) caused uv
  to greedily upgrade `transformers` past vllm 0.7.3's tolerance →
  runtime crash on training.
- **Fix**: New venv for any non-trivial new dep. Each venv has its
  own `requirements_<purpose>.txt`.
- **Memory rule**: [[venv-isolation-rule]] — process detail.

---

## 11. Related docs

- [docs/PROMPT_MIGRATION.md](PROMPT_MIGRATION.md) — separate workstream:
  11/15 production prompts migrated to YAML (extract, all 5 graphmert
  sub-step prompts, curriculum_verify, eval_models, rl_mcq). Completed
  same day; not bug-fixes but reduced future divergence surface.
- [docs/AUDIT_SFT_RL_HANDOFFS_2026-06-22.md](AUDIT_SFT_RL_HANDOFFS_2026-06-22.md) —
  proactive audit of remaining `curriculum + sft + rl` phases for the
  same family of bugs found here. 10 active crashers identified
  pre-emptively.
- Memory entries indexed in `[[brackets]]` above link back to per-fix
  detail in `~/.claude/projects/.../memory/`.

## 12. 2026-06-23 — Upstream `main` reconciliation

Jake pushed 3 commits to Princeton `main` between 13:55 and 14:11 ET on
2026-06-22. Branch state at review time: `orchestration` was 94 ahead
of `main`, 3 behind. Plain `git merge origin/main` would conflict in 3
files. Decision: skip the merge, hand-port the parts we want.

| SHA | What landed | Our handling |
|---|---|---|
| [1834992](https://github.com/scient-lab/neuro_SI_pipeline/commit/1834992) | README step 2.6 `--model_id` points at vLLM LLM | **Skip** — we don't sync upstream README; matches Jake's email |
| [41d7c8b](https://github.com/scient-lab/neuro_SI_pipeline/commit/41d7c8b) | `predict_tails.py` argparse CLI + in-file `AutoConfig.register("graphmert", ...)` + drop `TRANSFORMERS_CACHE=/tmp` + README step 2.6b | **Hand-port** to `2_graphmert/utils/predict_tails.py` preserving our normalization fixes (ef39998, 6d6f40d) and `architecture.py` constants (71bd627). Keep our env-var fallbacks. Rename `--output_root` → `--output_dir`. Add `--topk`/`--batch_size` flags. |
| [4d876bc](https://github.com/scient-lab/neuro_SI_pipeline/commit/4d876bc) | 10-bug audit fix bundle (README + `_load_kg` auto-detect in `calculate_hops.py` + `dataset_preprocessing_utils.py` + SLURM scripts) | **Cherry-port `_load_kg` only.** Inlined the parquet/csv auto-detect + source/target/description rename into both consumer sites. Skip Jake's README/SLURM/docstring changes (not load-bearing for us). |

### Conflicts dodged (vs. plain `git merge`)

| File | Their change | Our change | Why merge would muddle history |
|---|---|---|---|
| [2_graphmert/utils/predict_tails.py](../2_graphmert/utils/predict_tails.py) | argparse + AutoConfig.register inline | head normalization (6d6f40d), architecture.py extraction (71bd627) | Both edited top of file; resolver couldn't pick "Jake's CLI shape on top of our normalization" automatically |
| [2_graphmert/utils/dataset_preprocessing_utils.py](../2_graphmert/utils/dataset_preprocessing_utils.py) | `_load_kg` block | unique-IDs (c68e9b0), path stability (d0dd1d4), relation fallback (e5e3cad) | Both edited seed-KG load block from different angles |
| [2_graphmert/run_dataset_preprocessing.py](../2_graphmert/run_dataset_preprocessing.py) | docstring path update | path stability (d0dd1d4) | Cosmetic-only conflict; skipped |

### Net merge debt accepted

- 1834992 README docs fix (we don't sync upstream README)
- 4d876bc's SLURM script fixes (we run via `scripts/phases/*.sh`, not SLURM)
- 4d876bc's `run_dataset_preprocessing.py` docstring path update (cosmetic)

None of these affect runtime. Debt ≈ zero.

### Upstream bug to report back

[rl_training.slurm](../3_si_curriculum/slurm/rl_training.slurm) at
commit 4d876bc passes only `--model_name "${MODEL_NAME}"`, but
[rl_training.py:594](../3_si_curriculum/RL/rl_training.py#L594)
hard-raises `sft_checkpoint_path is required` when that flag is missing.
The SLURM job will crash at config parse. Worth flagging to Jake — fix
is one extra line: `--sft_checkpoint_path "${SFT_MERGED_PATH}" \`. Our
[scripts/phases/rl.sh](../scripts/phases/rl.sh) passes the correct
flag and is unaffected.

### 2026-06-23 edits applied

1. [2_graphmert/utils/predict_tails.py](../2_graphmert/utils/predict_tails.py) — Jake's CLI shape, preserve fork normalization
2. [3_si_curriculum/calculate_hops.py](../3_si_curriculum/calculate_hops.py) — add `_load_kg()` helper, route both loads through it, preserve empty-fallback
3. [2_graphmert/utils/dataset_preprocessing_utils.py](../2_graphmert/utils/dataset_preprocessing_utils.py) — inline parquet/csv autodetect + column rename at the seed-KG load site
4. [scripts/phases/graphmert.sh](../scripts/phases/graphmert.sh) — new `step_predict_tails_gm` between `predict_tails` and `validate_predictions`; skips silently when no trained checkpoint exists; toggle via `GRAPHMERT_PREDICT_TAILS_GM_REQUIRED=1`
5. C3 architectural note above superseded with Jake's clarification
