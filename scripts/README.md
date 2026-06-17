# scripts/

Orchestrator + extension points for the specialized-SLM pipeline.

## Quick reference

| Script | Where to run | What it does |
|---|---|---|
| `pipeline.sh` | local OR pod | Entry point. Parses `--domain` / `--profile` / `--platform` / `--phase` / `--step`, sources the matching venv, dispatches phases. |
| `launch_runpod.sh` | local only | POSTs a RunPod pod with secrets injected. Reads `.env.runpod` + `configs/profiles/<profile>.yaml::runpod`. |
| `runpod_bootstrap.sh` | pod only | First-run pod setup: apt install, install uv, clone repo, run `./setup.sh`, write `.env`. |
| `phases/<phase>.sh` | (sourced by pipeline.sh) | Per-phase wrapper. Sources the right venv, dispatches by step name. |
| `platforms/<platform>.sh` | (sourced by pipeline.sh) | Per-platform wrapper. Defines `exec_phase_on_platform`. |
| `lib/{common,venv}.sh` | (sourced helpers) | Logging, step filtering, venv activation. |

---

## Workflow 1 — Run locally on a workstation

```bash
# One-time setup
curl -LsSf https://astral.sh/uv/install.sh | sh   # install uv
./setup.sh                                          # create 3 venvs under .venvs/

# Run the pipeline
./scripts/pipeline.sh --profile smoke               # smallest end-to-end
./scripts/pipeline.sh --profile smoke --phase extract
./scripts/pipeline.sh --phase extract --step parse_pdf
./scripts/pipeline.sh --help                         # full flag reference
```

The pipeline reads from the repo's bundled configs/domains/prompts; no
environment variables are required for the standalone path.

---

## Workflow 2 — Run on RunPod

Two-step flow: **launch from local**, **bootstrap on pod**.

### Step 1 — Launch from your workstation

```bash
# One-time: copy the env template and fill in the secrets
cp .env.runpod.example .env.runpod
$EDITOR .env.runpod                                  # set RUNPOD_API_KEY, GITHUB_TOKEN,
                                                     # GEMINI_API_KEY, HF_TOKEN, etc.

# Dry-run first to see the POST body that will be sent (secrets masked)
./scripts/launch_runpod.sh --profile smoke --dry-run

# Launch a pod
./scripts/launch_runpod.sh                           # smoke profile (A4000, COMMUNITY)
./scripts/launch_runpod.sh --profile pilot           # A6000, SECURE, 150GB
./scripts/launch_runpod.sh --profile paper           # H100 80GB, SECURE, 300GB

# Override hardware on the fly (CLI > env > profile defaults)
./scripts/launch_runpod.sh --profile pilot \
    --gpu-type "NVIDIA H100 80GB HBM3" --disk-gb 250 --num-gpus 2
```

The launcher:
1. Reads `.env.runpod` for secrets (gitignored — never committed)
2. Reads `configs/profiles/<profile>.yaml::runpod` for GPU type, cloud type, disk, num_gpus
3. POSTs to `https://rest.runpod.io/v1/pods` with secrets injected as pod env vars
4. Prints the SSH info + the two bootstrap options for the pod

### Step 2 — Bootstrap the pod

Once the pod is reachable over SSH, the bootstrap script clones the repo,
sets up venvs, and writes `.env` from the injected secrets. **Two ways to
invoke it on the pod:**

**Option A — curl pipe (no checkout yet):**

```bash
# On the pod (ssh in first):
bash <(curl -sH "Authorization: token $GITHUB_TOKEN" \
            -H "Accept: application/vnd.github.v3.raw" \
            "https://api.github.com/repos/$GITHUB_REPO/contents/scripts/runpod_bootstrap.sh?ref=$GITHUB_BRANCH")
```

`$GITHUB_TOKEN`, `$GITHUB_REPO`, and `$GITHUB_BRANCH` are already exported
in the pod's environment (injected by the launcher). The curl pulls the
bootstrap script from GitHub and pipes it to bash — no manual clone needed.

**Option B — clone first, then run locally on the pod:**

```bash
# On the pod (ssh in first):
git clone https://${GITHUB_TOKEN}@github.com/${GITHUB_REPO}.git $SI_HOME
cd $SI_HOME && ./scripts/runpod_bootstrap.sh
```

Use Option B when you want to inspect the script before running it, or
when you're iterating on the bootstrap itself.

### Step 3 — Run the pipeline on the pod

```bash
cd $SI_HOME
source .env                                          # picks up GEMINI_API_KEY, HF_TOKEN, etc.
./scripts/pipeline.sh --profile $SI_PROFILE --platform runpod
```

`$SI_HOME` and `$SI_PROFILE` were injected by the launcher.

---

## Phase I/O reference

All paths below are relative to `$OUTPUT_BASE` (default: `$REPO_ROOT/outputs/`).
`OUTPUT_BASE` is settable via the env var of the same name.

### Phase 1 — extract  (venv: graphrag)

| Step | Reads | Writes | Format |
|---|---|---|---|
| `parse_pdf` | (no-op — we feed pre-extracted `.txt`) | — | — |
| stage-input | `$REPO_ROOT/$CORPUS_PATH/*.txt` | `graphrag/input/*.txt` | UTF-8 text, copied (not symlinked) |
| `chunk` (graphrag #1) | `graphrag/input/*.txt`, `graphrag/settings.yaml` | `graphrag/output/text_units` (HF storage) | parquet |
| `extract_triples` (graphrag #2+#3) | `graphrag/output/text_units` | `graphrag/output/extracted_graph_responses_<RELATION_SET>_<start>-<end>.json` | JSON (vLLM raw output per chunk) |
| `normalize` (graphrag #4) | `extracted_graph_responses_*.json` + `text_units` | `graphrag/output/entities`, `graphrag/output/relationships` | parquet (graphrag storage) |
| `cache` (graphrag #5) | `entities`, `relationships` | `graphrag/output/final_entities.parquet`, `final_relationships.parquet`, `kg_final.csv`, `kg_final.parquet`, `relation_counts_<RELATION_SET>_<ts>.txt` | parquet + CSV. `kg_final.{csv,parquet}` are columns `[head, relation, tail]` — the seed KG. |

### Phase 2 — validate

**No-op currently.** The 2-LLM consensus check is done inline in graphmert (fact_score) and curriculum (verify_questions) phases instead. See `phases/validate.sh` for the documented intent.

### Phase 3 — graphmert  (venv: graphmert)

| Step | Reads | Writes | Format |
|---|---|---|---|
| `tokenize` (1) | `graphrag/input/*.txt` | `graphmert/stable_tokenizer/`, `graphmert/tokenized_inputs/{train,val}_*/` | HF tokenizer + HF Dataset arrow |
| `preprocess` 2a (entity_discovery) | `graphmert/tokenized_inputs/train_*/` | `graphmert/entity_discovery/chunk_*/` | HF Dataset (entity head positions per chunk) |
| `preprocess` 2b (find_heads_positions) | `graphmert/entity_discovery/chunk_*/` | `graphmert/head_positions/` (directly — no sub-name) | HF Dataset |
| `preprocess` 3a (add_llm_relations) | `graphmert/head_positions/` + extract model | `graphmert/llm_relations/relations_all/` | HF Dataset |
| `preprocess` 3b (clean_llm_relations) | `graphmert/llm_relations/relations_all/` | `graphmert/llm_relations/relations_cleaned_train/`, `relations_cleaned_eval/` | HF Dataset |
| `preprocess` 4 (run_dataset_preprocessing) | `graphmert/llm_relations/relations_cleaned_*` + `graphrag/output/kg_final.csv` | `graphmert/dataset/relation_map.json`, `dataset/ready_for_training_{train,eval}/`, `mlm_cache/{train,validation}/ready_for_training/` | JSON + HF Dataset. mlm_cache is the canonical source MLM reads from. |
| `train_mnm` (5) | `graphmert/mlm_cache/{train,validation}/ready_for_training/` + `graphmert/args_mlm.resolved.yaml` | `graphmert/mlm_output/checkpoint-*/` | HF checkpoint dirs |
| `predict_tails` (6) | `mlm_output/checkpoint-*/` + `llm_relations/relations_cleaned_eval/` | `graphmert/predictions/predictions_shard_*.csv` | CSV |
| `validate_predictions` 7a (combine_tails) | `graphmert/predictions/*.csv` | `graphmert/combined/final_kg_all.csv`, `final_kg_scientific_only.csv`, `combined/expanded_triples.csv` | CSV |
| `validate_predictions` 7b (fact_score) | `graphmert/combined/expanded_triples.csv` + Gemini | `graphmert/final_kg/validated_triples.csv` | CSV (2-LLM-checked triples) |
| `expand_kg` | `graphmert/final_kg/validated_triples.csv` + `graphrag/output/kg_final.parquet` | `graphmert/final_kg/expanded_kg.parquet` | parquet (seed KG ∪ validated expansions) |

### Phase 4 — curriculum  (venv: si_curriculum)

| Step | Reads | Writes | Format |
|---|---|---|---|
| `path_traversal` (calculate_hops) | `graphmert/final_kg/validated_triples.csv` + `graphrag/output/kg_final.csv` | `curriculum/kg_manifest.json` | JSON (n-hop path manifest) |
| `prune_paths` | (configured via `hop_range` + `HUB_REMOVAL_PERCENTILE` inside generate_curriculum) | — | — |
| `generate_qa` (generate_curriculum) | `curriculum/kg_manifest.json` + Gemini API | `curriculum/curriculum.json` | JSON (Q&A items, one per path) |
| `validate_qa` (verify_questions) | `curriculum/curriculum.json` + Gemini (2 calls per item) | `curriculum_verified/curriculum_verified.json` | JSON (filtered) |
| `assemble_curriculum` | (no-op) | — | — |

### Phase 5 — sft  (venv: si_curriculum)

| Step | Reads | Writes | Format |
|---|---|---|---|
| `prepare_data` | `curriculum_verified/curriculum_verified.json` | `sft_dataset/` | HF Dataset (tokenized SFT prompts/responses) |
| `train_lora` | `sft_dataset/` + base model (HF download) | `sft_checkpoints/checkpoint-*/` | HF checkpoint (LoRA adapters) |
| `merge_lora` | `sft_checkpoints/checkpoint-*/` | `sft_checkpoints/checkpoint-*/merged_final_model/` | Merged HF model (base + LoRA) |
| `eval_sft` | (no-op — operator runs `eval_models.py` separately) | — | — |

### Phase 6 — rl  (venv: si_curriculum)

| Step | Reads | Writes | Format |
|---|---|---|---|
| `setup_reward` (data_prep) | `curriculum_verified/curriculum_verified.json` | `rl_dataset/` | HF Dataset (GRPO prompts + reward signals) |
| `train_grpo` (rl_training) | `sft_checkpoints/checkpoint-*/merged_final_model/` + `rl_dataset/` + `3_si_curriculum/RL/deepspeed_config.json` | `rl_checkpoints/checkpoint-*/` | HF checkpoint |
| `eval_rl` | (no-op — operator runs `eval_models.py` separately) | — | — |

### Cross-phase artifacts (not phase-specific)

| Path | Written by | Read by |
|---|---|---|
| `wandb_logs/` | sft + rl | W&B background uploader (if `WANDB_API_KEY` set) |
| `args_mlm.resolved.yaml` | graphmert phase setup (envsubst from template) | run_dataset_preprocessing.py + run_mlm.py |

---

## Logs

Currently **stdout only** — `log_info` / `log_error` from `lib/common.sh` print to terminal, nothing is captured to a file. If the pod dies mid-run, the logs die with it. Two practical patches you can apply:

```bash
# Capture full pipeline log to disk (synchronous, ~zero overhead):
./scripts/pipeline.sh --profile pilot --platform runpod 2>&1 | tee outputs/pipeline.log
```

W&B (training logs only — not the orchestration) IS asynchronous if `WANDB_API_KEY` is set: it uploads to W&B cloud in a background process while training continues. Local mirror lives in `outputs/wandb_logs/`.

**TODO:** plumb `tee outputs/logs/<phase>.log` into pipeline.sh per phase so we always have a phase-by-phase log file, and include the logs dir in the output S3 sync.

## Output → S3 sync (NOT implemented yet)

Only **input/corpus** sync is built. Outputs live on the pod's ephemeral disk and **disappear when the pod is shut down**. Discussed three approaches (see history): per-phase sync, end-of-pipeline sync, continuous mirror. Recommended next step:

```bash
# scripts/data_prep/sync_outputs.sh  (companion to sync_corpus.sh)
# Push the whole outputs/ dir to s3://${S3_URI}/runs/<run-id>/outputs/ at
# the end of each phase, with --exclude 'graphrag/cache/*' --exclude 'input/*'.
# run-id = ${SI_PROFILE}-$(date -u +%Y%m%d-%H%M%S)-${git_short_sha}
```

Once built, hook it at end of each `phases/*.sh` so a mid-run crash doesn't lose progress (per-phase resilience) PLUS one final sync at end of `pipeline.sh`.

---

## Extension points

### Add a new phase

1. Copy `phases/extract.sh` as the template.
2. Set `STEPS=(...)` for the phase's ordered step list.
3. Pick the matching venv with `source_venv graphrag|graphmert|si_curriculum`.
4. Add the phase name to `ALL_PHASES` in `pipeline.sh`.

### Add a new platform

1. Copy `platforms/local.sh` as the template.
2. Define `exec_phase_on_platform` with any platform-specific bootstrap
   (mounts, env, scratch directories) before the inner `bash` call.
3. Add `configs/platforms/<name>.yaml` with hardware/storage settings.

### Add a new RunPod scenario

Edit the relevant `configs/profiles/<profile>.yaml` and add or update the
`runpod:` block. Keys: `gpu_type`, `cloud_type` (COMMUNITY or SECURE),
`disk_gb`, `num_gpus`. The launcher reads these by default; CLI flags
still override on a per-launch basis.

---

## Files in this directory

```
scripts/
├── pipeline.sh             # orchestrator entry point
├── launch_runpod.sh        # workstation: POST pod to RunPod API
├── runpod_bootstrap.sh     # pod: clone + setup.sh + .env
├── phases/
│   ├── extract.sh          # phase 1
│   ├── validate.sh         # phase 2
│   ├── graphmert.sh        # phase 3
│   ├── curriculum.sh       # phase 4
│   ├── sft.sh              # phase 5a
│   └── rl.sh               # phase 5b
├── platforms/
│   ├── local.sh            # workstation / Princeton on-prem
│   ├── runpod.sh           # RunPod pod (after bootstrap)
│   ├── aws.sh              # EC2 / SageMaker
│   └── princeton.sh        # delegates to local.sh
└── lib/
    ├── common.sh           # log_info / log_warn / log_error / step_enabled
    └── venv.sh             # source_venv <name>
```
