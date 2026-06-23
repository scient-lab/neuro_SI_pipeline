# Bug Failure Cases (2026-06-22)

Where today's 29 bugs would actually break vs where they would not.
Companion to [docs/EXTRACT_GRAPHMERT_BUGS.md](EXTRACT_GRAPHMERT_BUGS.md)
(the full ledger) and [docs/AUDIT_SFT_RL_HANDOFFS_2026-06-22.md](AUDIT_SFT_RL_HANDOFFS_2026-06-22.md)
(the SFT/RL audit).

## TL;DR

Bugs split four ways by failure mode. Counts are approximate (some bugs
straddle two cases):

| Case | What it is | Count | Princeton's env? | Our env? | Where does the fix go? |
|------|-----------|------:|-----------------:|---------:|------------------------|
| A | Pure environment divergence | ~5 | works | breaks | fork-side config or producer step |
| B | Genuine code bugs (crash anywhere) | ~9 | breaks | breaks | code patch, applies to both sides |
| C | External time/version drift | ~3 | breaks today | breaks today | swap to a current model/repo/SDK |
| D | Fork-introduced modifications | ~10 | works | breaks | revert to upstream OR finish the fork modification |
| **Total** | | **~27-29** | | | (the spread is because a few span two cases) |

Headline takeaway, free of personality:

> The original upstream code was likely run end-to-end at Princeton on
> their bundled data, conda env, and SLURM cluster. After that, it was
> ported into this fork (commit `33823cb`) and re-merged twice (most
> recently `2d7b782`) with substantial modifications. Neither side ran
> the integrated chain end-to-end before today. So Case A bugs are real
> env-divergence (Princeton hides them); Cases B and D are bugs that
> would crash at both sites if either side ran the integrated chain.

Forward fix is one engineering-process change: every code merge (upstream
or fork-side) requires a full `extract -> graphmert -> curriculum -> sft
-> rl` smoke run before it is considered "landed." The gate catches every
case below before it bleeds an engineering day.

---

## Case A: Pure environment divergence

Princeton's environment makes them not fire. Our environment surfaces
them. The code itself isn't wrong - it just assumed a setup we don't
have.

### A.1: SLURM-only learning rate list

- **What's wrong**: `mlm_utils.py:90-91` indexes a YAML
  `learning_rates: [0.0004, 0.0005, ...]` list by `SLURM_ARRAY_TASK_ID`.
- **Princeton's setup**: launches via `sbatch --array=0-3`. `is_slurm`
  is True. Each parallel job picks a different LR.
- **Our setup**: RunPod single-node. `SLURM_ARRAY_TASK_ID` is unset.
  `is_slurm` is False. The list is silently ignored. HF default
  `learning_rate=5e-5` wins.
- **Fix**: scalar `learning_rate: !!float 0.0004` in YAML (works in any
  env, including SLURM).
- **Reference**: memory `[mlm-lr-scaling-bug]`, ledger entry H5.

### A.2: Bundled `data_kg/` files (DDB KG)

- **What's wrong**: `QAGenerator.__init__` requires `vocab.txt`,
  `ddb.graph`, and `vocab_freq.json` in a `kg_dir` directory.
- **Princeton's setup**: these files are checked into the upstream repo
  at `bottom-up-superintelligence/curriculum_generator/data_kg/`. They
  ship as part of the source tree.
- **Our setup**: the equivalent neuroscience files don't exist anywhere
  in the orchestration. No producer step writes them. The fork tried to
  call this with `kg_dir=None` and crashed at construction.
- **Fix**: made `PathGenerator` lazy-init, plus added
  `generate_from_path(path_data)` that bypasses it entirely. Curriculum
  now generates from the seed KG directly.
- **Reference**: ledger entry #2 + the calculate_hops fallback.

### A.3: `vllm` vs `torch` version conflict (uv strict resolver)

- **What's wrong**: `vllm==0.7.3` hard-requires `torch==2.5.1`.
  `3_si_curriculum/requirements.txt` pinned `torch==2.4.0+cu121`.
- **Princeton's setup**: conda's forgiving resolver picks any
  vllm/torch combination that resolves transitively. Or their conda
  env never had this exact conflict because vllm came in transitively.
- **Our setup**: uv's strict resolver refuses to install. Error
  surfaces in 2 seconds at install time, not as a runtime crash later.
- **Fix**: bumped torch to 2.5.1+cu121 to match vllm 0.7.3 + match
  `2_graphmert/requirements.txt`.
- **Reference**: ledger entry #13.

### A.4: TailSlot collator vs `mlm_sbo=true`

- **What's wrong**: upstream `args_mlm.yaml` has `mlm_sbo=true`. Upstream
  uses `GraphMertDataCollator` (Cython) which emits SBO `pairs` field
  in every batch.
- **Princeton's setup**: the Cython collator runs. `pairs` is populated.
  Model's SBO head consumes `pairs`. Works.
- **Our setup**: our fork's `mlm_utils.py:239` swaps to
  `GraphMertTailSlotDataCollator` (Python, no Cython, no SBO). `pairs`
  is never emitted. Model expects `pairs`, gets None, crashes.
- **Fix**: `mlm_sbo=false` in our `args_mlm.yaml`. Aligns config with
  the collator we actually chose.
- **Reference**: ledger entry C1.

### A.5: CUDA driver vs torch wheel mismatch on certain RunPod images

- **What's wrong**: RunPod base image `runpod/pytorch:1.0.2-cu1281-torch280-...`
  landed on a host with driver API 12.4 - too old for torch built
  against cu128. Silent CPU fallback.
- **Princeton's setup**: Della cluster GPU drivers are kept synchronized
  with their compute stack. Mismatch doesn't happen.
- **Our setup**: RunPod base images vary per-pod-tier. Mismatch is
  visible if you check `torch.cuda.is_available()`.
- **Fix**: use cu124 base image or force-reinstall torch from cu124
  wheels.
- **Reference**: memory `[graphmert-runpod-gotchas]`, ledger entry H8.

**Case A summary**: ~5 bugs. These are real environment divergence.
Fix is to add a producer step (vocab.txt etc.), update config (mlm_sbo,
learning_rate, torch pin), or change RunPod base image.

---

## Case B: Genuine code bugs, would crash anywhere

These are wrong code regardless of operating system, cluster type,
container image, or model family. They would crash at Princeton if
Princeton actually ran them.

### B.1: `combine_tails.py` filename filter `"exploded"`

- **What's wrong**: `combine_tails.py:96` filters CSV files by
  `"exploded" in f`. No producer in the repo writes a file with
  `"exploded"` in its name. `predict_tails_llm.py:223` writes
  `predictions_shard{N}_of{M}.csv`.
- **Where it'd break**: at Princeton too, if `combine_tails` was ever
  run against `predict_tails_llm.py` output. Latent because no Princeton
  flow actually chained these two scripts.
- **Fix**: ledger entry W1.

### B.2: `graphmert.sh` -> `fact_score.py` filename

- **What's wrong**: shell passes `expanded_triples.csv` to
  `fact_score.py`. No producer writes that filename.
- **Where it'd break**: nowhere has a producer. Crashes anywhere.
- **Fix**: ledger entry W2.

### B.3: `merge_kgs.py` parquet-only loader

- **What's wrong**: `_load_parquet` is hardcoded `pd.read_parquet`. The
  caller passes `validated_triples.csv`. pyarrow rejects with magic-byte
  error.
- **Where it'd break**: anywhere a CSV is passed to a function that
  hardcodes parquet. Doesn't depend on env.
- **Fix**: ledger entry W3.

### B.4: `merge_kgs.py` requires `source`/`target` columns

- **What's wrong**: function checks `for c in ["source", "target"]: if
  c not in df.columns: raise KeyError`. Input CSV has `head`/`relation`/
  `tail`. KeyError.
- **Where it'd break**: anywhere a head/tail CSV is fed in.
- **Fix**: ledger entry W4.

### B.5: `combine_tails.py` Qwen3 `<think>` block parser

- **What's wrong**: parser does `text.startswith("yes")` on raw response.
  Qwen3's response starts with `<think>...reasoning...</think>YES`.
  `<think>...` never starts with `yes`. 100% rejection.
- **Where it'd break**: anywhere Qwen3 with thinking is the LLM. Doesn't
  depend on env. If Princeton ever ran combine_tails through Qwen3,
  they'd hit this too.
- **Fix**: ledger entry L1.

### B.6: `predict_tails_llm.py` function-local import shadow (latent)

- **What's wrong**: `main()` has `from pipeline_config import
  get_phase_param` inside the function. Python treats the name as local
  for the entire function. If any code earlier in `main()` references
  it, `UnboundLocalError`.
- **Where it'd break**: latent. Doesn't fire today because no earlier
  reference exists. Would fire at Princeton too if anyone added one.
- **Fix**: ledger entry D2.

### B.7: `combine_tails.py` function-local import shadow

- **What's wrong**: same pattern as B.6 but the shadow DID fire. The
  earlier `get_phase_param` reference at line 189 hit UnboundLocalError
  on every run.
- **Where it'd break**: at Princeton too, if they ran combine_tails.
- **Fix**: ledger entry D3.

### B.8: `rl.sh` -> `rl_training.py` `--model_name` flag mismatch

- **What's wrong**: shell passes `--model_name`. The script's
  `rl_training.py:589` reads `config.sft_checkpoint_path` (different
  field name). `--model_name` is set but never read.
- **Where it'd break**: anywhere - it's a typo'd flag name. Crashes at
  `from_pretrained("")`.
- **Fix**: audit #8 / ledger entry rl.train_grpo.

### B.9: `rl/data_prep.py` double-preprocessing

- **What's wrong**: data_prep writes a processed dataset (drops
  `question_and_explanation` column). rl_training.py then re-runs
  `preprocess_grpo_dataset` on the same path. Second call raises
  `KeyError: 'question_and_explanation'` because the column was already
  consumed.
- **Where it'd break**: anywhere the two scripts are chained.
- **Fix**: audit #10 / data_prep "rl" mode no longer chains preprocess.

### B.10: `MLM_CHECKPOINT` default passes GraphMERT path to vLLM

- **What's wrong**: `graphmert.sh:162` defaults
  `MLM_CHECKPOINT=$GRAPHMERT_DIR/checkpoints/best`. vLLM tries to load
  it as a causal LM. BERT-style MLM is not a supported architecture.
- **Where it'd break**: anywhere - vLLM literally cannot load BERT.
- **Fix**: ledger entry C3 (added `models.predict_tails: Qwen/Qwen3-14B`).

**Case B summary**: ~9-10 bugs. These were code defects from day one of
the integrated pipeline. They would crash at Princeton if Princeton ran
the same chain. The fact that today is the first time they fired means
the integrated chain has never been tested end-to-end at either site.

---

## Case C: External time/version drift

The world moved after the original code was written.

### C.1: `gemini-2.0-flash` returns HTTP 404

- **What happened**: Google deprecated `gemini-2.0-flash` for new users.
  Old API key holders may still have access; new ones get 404 NOT_FOUND.
- **When was code written**: before 2.5-flash existed.
- **Where it'd break**: anywhere today, using a recently-issued API key.
  Same fault for Princeton if they pulled a new API key.
- **Fix**: swap to `gemini-2.5-flash`. Ledger entry #15.

### C.2: `text-embedding-004` deprecated (Phase C only, didn't hit today)

- **What happened**: same family - Google deprecated this model after
  the code was written.
- **Reference**: memory `[graphmert-runpod-gotchas]`.

### C.3: `openai/gpt-oss-20b` uses MXFP4 quantization

- **What happened**: OpenAI released gpt-oss with MXFP4 quantization
  after vllm 0.7.3 (which has no MXFP4 support).
- **Where it'd break**: anywhere using vllm 0.7.3 with this model. Same
  fault at Princeton if they pinned the same vllm version.
- **Fix**: ledger entry C4. Swapped to Qwen3-14B.

### C.4: `mistralai/Mistral-Nemo-12B` HF repo doesn't exist

- **What happened**: shorthand "Mistral-Nemo-12B" was never a real
  HuggingFace repo id. The real ids are
  `mistralai/Mistral-Nemo-Base-2407` and `mistralai/Mistral-Nemo-Instruct-2407`.
  Likely a transcription error from a paper or blog.
- **Where it'd break**: anywhere using this id. HF returns 404.
- **Fix**: ledger entry C5.

**Case C summary**: ~3 bugs. External world moved or had bad data. Not
specific to any environment. Same fault at Princeton if they tried
today with current credentials.

---

## Case D: Fork-introduced modifications

Modifications made AFTER the original code was ported into this fork
broke previously-working upstream behavior. These are not Jake's bugs;
they are bugs that arose because the fork team modified upstream code
without integration-testing the result.

### D.1: `data_prep.py` simplified (lost DatasetDict + text column)

- **What happened**: upstream `bottom-up-superintelligence/data/tokenization.py`
  writes a `DatasetDict({"train": ...})` with a `text` column. The fork
  rewrote this as `3_si_curriculum/training/data_prep.py` to write a
  flat `Dataset` with `input_ids`/`attention_mask` only.
- **Why**: probably an attempt to simplify; lost the train/test split
  and the text column in the process.
- **Where it'd break**: anywhere - trainer.py indexes `dataset['train']`
  and expects `text` field. Fork's simplified version doesn't provide
  either.
- **Where it would NOT break**: Princeton's setup - upstream's
  tokenization.py emits the right shape.
- **Fix**: audit #4 + #5. Reverted toward upstream's design (DatasetDict
  + text column + apply_chat_template).

### D.2: `trainer.py` hardcoded DeepSeek `<｜Assistant｜>` token

- **What happened**: upstream `trainer.py:186-187` uses Qwen ChatML
  format. The fork's version replaces that with a hardcoded lookup for
  DeepSeek's special token.
- **Why**: someone tried to support DeepSeek-R1-Qwen3-8B as base_sft.
  But the fork's `configs/default.yaml::models.base_sft` is
  `Qwen/Qwen3-14B`, not DeepSeek.
- **Where it'd break**: anywhere using Qwen base_sft. Token not in vocab.
- **Where it would NOT break**: at Princeton if they used DeepSeek as
  base_sft.
- **Fix**: audit #6. Auto-detect tokenizer family.

### D.3: Generate_from_path called but never defined

- **What happened**: fork's `generate_curriculum.py:154` calls
  `generator.generate_from_path(path_data)`. The QAGenerator class
  doesn't define this method. Upstream's generate_curriculum.py uses a
  different pattern (`qa_gym.generator.vocab_freq[...]`).
- **Why**: fork added the call planning to add the method later, never
  did. Or refactored away from upstream's pattern without finishing.
- **Where it'd break**: anywhere. AttributeError on first iteration.
- **Where it would NOT break**: at Princeton if they kept upstream's
  generate_curriculum.py unmodified.
- **Fix**: audit #1. Added the method.

### D.4: `expanded_triples.csv` invented filename

- **What happened**: fork's `scripts/phases/graphmert.sh` invents a
  filename for fact_score input that no producer writes.
- **Why**: graphmert.sh is fork-only (Princeton doesn't have pipeline.sh
  shell orchestration). Someone wrote the shell wrappers without
  matching them to actual producer outputs.
- **Where it'd break**: anywhere running fork's graphmert.sh.
- **Where it would NOT break**: at Princeton (they don't have
  graphmert.sh).
- **Fix**: ledger entry W2.

### D.5: `rl.sh` `--model_name` typo

- **What happened**: same pattern as D.4. rl.sh wraps rl_training.py
  with a wrong flag name. rl.sh is fork-only.
- **Fix**: audit #8.

### D.6: `rl.sh` double-preprocessing chain

- **What happened**: rl.sh chains data_prep.py then rl_training.py,
  both of which call `preprocess_grpo_dataset`. Princeton runs them
  separately, never both.
- **Fix**: audit #10.

### D.7: `curriculum.sh` broken `require_env || export` shell logic

- **What happened**: fork's curriculum.sh has
  `require_env GEMINI_API_KEY || export GOOGLE_API_KEY=...`. The `||`
  branch is unreachable in both directions (require_env exits 1 on
  missing var; the export only runs on require_env success which is...
  rare to never).
- **Why**: fork-only shell wrapper. Logic error introduced when this
  was written.
- **Fix**: ledger entry #11. Direct unconditional export.

### D.8: `calculate_hops.py` empty kg_path fallback (added today)

- **What happened**: upstream calculate_hops.py was written assuming
  the kg_path always has rows. At smoke scale, graphmert produces 0
  validated triples, so kg_path is empty.
- **Why this is "fork-introduced"**: upstream never had a "smoke" scale
  that produces 0 validated triples - they ran at paper scale.
- **Fix**: ledger entry #14. Added empty-fallback to use seed KG.

### D.9: `mlm_sbo=true` config not aligned with TailSlot collator

- **What happened**: see Case A.4 above. The CONFIG value is upstream's;
  the COLLATOR choice is fork's. The mismatch is fork-introduced.
- **Fix**: ledger entry C1.

### D.10: Import shadows in fork's defensive code

- **What happened**: someone added `from pipeline_config import
  get_phase_param` inside `main()` of multiple files (defensive
  re-import). This introduced Python scoping bugs that didn't exist in
  upstream.
- **Fix**: ledger entries D2, D3. Removed the in-function imports.

**Case D summary**: ~10 bugs. Fork-introduced modifications broke
previously-working upstream behavior. These are not Jake's bugs - they
are the fork team's own. The pattern: modifications made without
integration-testing the result.

---

## Combined view: all 29 bugs

| Bug | Case | Why it didn't fire at Princeton | Why it fires here |
|----:|:----:|---------------------------------|-------------------|
| E1 (Python.h apt) | A | conda installs python3-dev | apt-only on RunPod |
| C1 (mlm_sbo=true) | A+D | upstream collator emits SBO | fork uses TailSlot, no SBO |
| C2 (eval_loss key) | B | always fires if eval N=2 | smoke eval is N=2 |
| C3 (MLM_CHECKPOINT default) | B | README:365 + predict_tails_llm.py mismatch exists on Princeton main too; if anyone ran their own README:365 literally, they'd hit `ValueError: model type 'graphmert' not recognized` from HF AutoConfig | empirically verified via vLLM repro (see EXTRACT_GRAPHMERT_BUGS.md C3); fork's shell only perpetuates the upstream-broken invocation |
| C4 (gpt-oss-20b mxfp4) | C | vllm pinned same way | hits everyone with vllm 0.7.3 |
| C5 (Mistral-Nemo-12B 404) | C | wrong id everywhere | 404 everywhere |
| D1 (None-guard `pairs`) | B+D | upstream collator always emits pairs | fork collator doesn't |
| D2 (predict_tails import shadow) | B | latent; never fires unless used | latent |
| D3 (combine_tails import shadow) | B | would crash at Princeton too | crashes here |
| W1 (predictions_shard filter) | B | filename mismatch in shared code | filename mismatch in shared code |
| W2 (expanded_triples.csv) | D | no graphmert.sh upstream | fork wired wrong filename |
| W3 (parquet vs CSV) | B | would crash at Princeton too | crashes here |
| W4 (source/target vs head/tail) | B | column mismatch in shared code | column mismatch in shared code |
| L1 (Qwen3 think parser) | B | would crash with Qwen3 anywhere | crashes here |
| H1 (Cython missing pip) | A | conda picks up Cython | uv strict; explicit pin needed |
| H2 (networkx + vllm missing) | A | conda picks up networkx | uv strict |
| H3 (`_norm_head` Fix A.2) | D | upstream may have different inner sites | fork's bug |
| H4 (diabetes prompts) | D | upstream's prompts are medical, intended | fork's domain is neuroscience |
| #1 (generate_from_path missing) | D | upstream code is unmodified | fork modified call site, not method |
| #2 (kg_dir requirement) | A | bundled DDB KG files exist | fork has no producer |
| #3 (curriculum.json filename) | D | upstream's curriculum.sh expects timestamped name (TBD) | fork's curriculum.sh expects stable name |
| #4 (DatasetDict missing) | D | upstream emits DatasetDict | fork's data_prep flattened it |
| #5 (text column missing) | D | upstream emits text column | fork's data_prep removed it |
| #6 (DeepSeek vs Qwen ChatML) | D | upstream uses Qwen ChatML | fork modified to DeepSeek |
| #7 (chat template mismatch) | D | sister of #6 | sister of #6 |
| #8 (--model_name flag) | D | no rl.sh upstream | fork shell typo |
| #9 (dataset_path placeholder) | B | latent; ad-hoc-run-only | latent |
| #10 (double preprocess) | D | upstream doesn't chain | fork's rl.sh chains |
| #11 (GEMINI_API_KEY shell) | D | no curriculum.sh upstream | fork shell logic error |
| #12 (google-generativeai) | D | conda installs the new SDK separately | uv strict; explicit pin needed |
| #13 (torch/vllm) | A | conda transitive resolution | uv strict resolver |
| #14 (calculate_hops empty fallback) | D | upstream runs at paper scale | smoke scale produces 0 triples |
| #15 (gemini-2.0-flash 404) | C | same fault at Princeton | external deprecation |

---

## What this means

### Honest attribution

| Question | Answer based on the data |
|----------|--------------------------|
| Did Princeton run their `bottom-up-superintelligence` end-to-end? | Plausibly yes, on their bundled DDB data + their conda env + Della SLURM. Case A bugs are real env divergence. |
| Did Princeton run the INTEGRATED neuro_SI_pipeline end-to-end? | No. ~9 Case B bugs would crash for them too. |
| Did the fork team run our modified code end-to-end before today? | No. ~10 Case D bugs are our own modifications that broke previously-working upstream code. |
| Was today's failure pattern unique to our environment? | Only ~5 bugs (Case A). The other ~24 would have broken anywhere. |

### Forward fix

Single engineering-process change applied to both sides:

1. **Smoke-validation gate before every merge.** Every code merge
   (upstream OR fork-side) requires a full `extract -> graphmert ->
   curriculum -> sft -> rl` smoke run before it is considered "landed."
2. **Same gate on upstream pulls.** When Princeton ships new code, run
   the same smoke chain BEFORE merging. Catches Case B and Case D bugs
   before they bleed an engineering day.
3. **Documented env divergence list.** Maintain a list of Case A items
   (SLURM, DDB KG, conda/uv, base model) as known-divergent and
   actively monitored.

Cost per merge with the gate: estimated ~30 minutes. Cost today without
it: ~10 hours. The math is straightforward.

### For external communication (CEO, partners)

Drop "Jake threw it over the wall" framing. The data shows it was
roughly:

- 5 bugs are env divergence (Princeton's setup hides them)
- 9 bugs are code defects that would crash at both sites
- 3 bugs are external time/version drift (Google deprecated, etc.)
- 10 bugs are fork-introduced modifications (our own work)
- Roughly half of today's bugs are fork-introduced, not Princeton-shipped

Neutral framing for a status update:

> Today surfaced 29 bugs across the integrated pipeline. Distribution:
> ~5 environment-divergence (Princeton's SLURM/conda/bundled-KG setup
> hides them), ~9 code defects that would crash at both sites (meaning
> the integrated chain has never been tested end-to-end at any site),
> ~3 external version drift (deprecated model IDs and HF repos), and
> ~10 fork-introduced modifications. Forward fix is a single process
> change: smoke-validation gate before every merge, applied symmetrically
> to upstream pulls and fork-side changes. Estimated cost per merge: ~30
> minutes. Estimated time saved per skipped gate: full engineering day.

---

## Related docs

- [docs/EXTRACT_GRAPHMERT_BUGS.md](EXTRACT_GRAPHMERT_BUGS.md): full
  ledger of every bug with file:line refs and fix patches.
- [docs/AUDIT_SFT_RL_HANDOFFS_2026-06-22.md](AUDIT_SFT_RL_HANDOFFS_2026-06-22.md):
  preemptive audit of SFT/RL phases that surfaced 10 of these bugs.
- [docs/PROMPT_MIGRATION.md](PROMPT_MIGRATION.md): separate workstream
  on YAML prompt migration (orthogonal to today's incident).
