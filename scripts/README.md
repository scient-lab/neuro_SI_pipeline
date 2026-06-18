# scripts/

Orchestrator + extension points for the specialized-SLM pipeline.

## Quick reference

| Script | Where to run | What it does |
|---|---|---|
| `pipeline.sh` | local OR pod | Entry point. Parses `--domain` / `--profile` / `--platform` / `--phase` / `--step`, sources the matching venv, dispatches phases. |
| `logs.sh` | local OR pod | View per-run / per-phase / per-step logs. Triage failures via `--error`. See [Viewing logs](#viewing-logs). |
| `runpod/launch.sh` | local only | POSTs a RunPod pod with secrets injected. Reads `.env.runpod` + `configs/profiles/<profile>.yaml::runpod`. |
| `runpod/bootstrap.sh` | pod only | First-run pod setup: apt install, install uv, clone repo, run `./setup.sh`, write `.env`. |
| `phases/<phase>.sh` | (sourced by pipeline.sh) | Per-phase wrapper. Sources the right venv, dispatches by step name. |
| `platforms/<platform>.sh` | (sourced by pipeline.sh) | Per-platform wrapper. Defines `exec_phase_on_platform`. |
| `lib/{common,venv,manifest}.sh\|.py` | (sourced helpers) | Logging, step filtering, venv activation, manifest mutations. |
| `data_prep/{sync_corpus,sync_outputs}.sh` | local + pod | S3 ↔ local sync for input corpus and run outputs. |

---

## Workflow 1 — Run locally on a workstation

```bash
# One-time setup
curl -LsSf https://astral.sh/uv/install.sh | sh   # install uv
./setup.sh                                          # create 3 venvs under .venvs/

# Run the pipeline (long-running invocations use nohup so SSH disconnects /
# closed terminals don't kill the run; per-phase logs at outputs/logs/<RUN_ID>/)
nohup ./scripts/pipeline.sh --profile smoke              > nohup.out 2>&1 &  # smallest end-to-end
nohup ./scripts/pipeline.sh --profile smoke --phase extract > nohup.out 2>&1 &
nohup ./scripts/pipeline.sh --phase extract --step parse_pdf > nohup.out 2>&1 &

# Track / inspect
tail -f nohup.out                                   # orchestrator stdout
tail -f outputs/logs/*/extract.log                  # latest phase log
pgrep -af pipeline.sh                               # pid + cmdline

# Interactive (no nohup needed)
./scripts/pipeline.sh --help                        # full flag reference
./scripts/pipeline.sh --list                        # phases + steps
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
./scripts/runpod/launch.sh --profile smoke --dry-run

# Launch a pod
./scripts/runpod/launch.sh                           # smoke profile (A4000, COMMUNITY)
./scripts/runpod/launch.sh --profile pilot           # A6000, SECURE, 150GB
./scripts/runpod/launch.sh --profile paper           # H100 80GB, SECURE, 300GB

# Override hardware on the fly (CLI > env > profile defaults)
./scripts/runpod/launch.sh --profile pilot \
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
            "https://api.github.com/repos/$GITHUB_REPO/contents/scripts/runpod/bootstrap.sh?ref=$GITHUB_BRANCH")
```

`$GITHUB_TOKEN`, `$GITHUB_REPO`, and `$GITHUB_BRANCH` are already exported
in the pod's environment (injected by the launcher). The curl pulls the
bootstrap script from GitHub and pipes it to bash — no manual clone needed.

**Option B — clone first, then run locally on the pod:**

```bash
# On the pod (ssh in first):
git clone https://${GITHUB_TOKEN}@github.com/${GITHUB_REPO}.git $SI_HOME
cd $SI_HOME && ./scripts/runpod/bootstrap.sh
```

Use Option B when you want to inspect the script before running it, or
when you're iterating on the bootstrap itself.

### Step 3 — Run the pipeline on the pod

```bash
cd $SI_HOME
# pipeline.sh auto-sources .env (set -a wrapper). Use nohup + & so the run
# survives SSH disconnect; per-phase logs at outputs/logs/<RUN_ID>/<phase>.log.
nohup ./scripts/pipeline.sh --profile $SI_PROFILE --platform runpod > nohup.out 2>&1 &
tail -f nohup.out                                    # follow live
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

## Run identity (RUN_ID + manifest)

Every pipeline.sh invocation generates a `RUN_ID` of the form:

```
<UTC-timestamp>-<profile>-<git-short-sha>
e.g. 20260617-141523-pilot-a1b2c3d
```

Exported to all phases. Lists chronologically; embedded profile + sha make it
grep-friendly across many runs.

## Run manifest (live status — API-consumable)

`$OUTPUT_BASE/run_manifest.json` is a **live status document**, not a
write-once record: it is created at run start and rewritten at every phase/step
transition by the stdlib-only `scripts/lib/manifest.py` (atomic write + flock,
so a process reading it mid-run — e.g. an API polling the S3 copy — never sees a
half-written file). It has two halves:

- **`meta`** — STATIC catalog, identical for every run. The `status_enum`, the
  `timestamp_format`, and the canonical ordered list of phases, each with its
  ordered steps + descriptions (parsed straight from `phases/<phase>.sh`). A
  consumer reads this once to learn the whole pipeline shape, including phases/
  steps that were *not* selected this run.
- **`run`** — PER-RUN state: which phases/steps were selected, their `status`
  (`pending → running → completed | failed | skipped`), tz-aware start/end
  timestamps **per phase AND per step**, exit codes, and per-step `log_file`
  (+ `cw_log_stream` when CloudWatch is on).

```json
{
  "schema_version": "1.0",
  "meta": {
    "status_enum": ["pending", "running", "completed", "failed", "skipped"],
    "timestamp_format": "RFC3339 / ISO-8601 with timezone offset (e.g. 2026-06-17T14:15:23+00:00)",
    "phases": [
      { "name": "extract", "description": "Build seed KG …",
        "steps": [ {"name": "chunk", "description": "Chunk text…"} ] }
    ]
  },
  "run": {
    "run_id": "20260617-141523-pilot-a1b2c3d",
    "status": "running",
    "domain": "neuroscience", "profile": "pilot", "platform": "runpod",
    "git_sha": "a1b2c3d", "git_branch": "orchestration", "step_filter": "all",
    "started_at": "2026-06-17T14:15:23+00:00", "finished_at": null,
    "current_phase": "graphmert",
    "selected_phases": ["extract", "graphmert"],
    "phases": [
      { "name": "extract", "status": "completed",
        "started_at": "…+00:00", "finished_at": "…+00:00",
        "exit_code": 0, "log_file": "outputs/logs/<run_id>/extract.log",
        "steps": [
          { "name": "chunk", "status": "completed",
            "started_at": "…+00:00", "finished_at": "…+00:00",
            "exit_code": 0, "log_file": "outputs/logs/<run_id>/extract/chunk.log",
            "cw_log_stream": null } ] } ]
  }
}
```

### Completion signals for an API / downstream job

Two integration modes, both reaching S3 alongside the outputs:

- **Rich poll** — read `run_manifest.json`; branch on `run.status` and per-phase/
  per-step `status`. Good for progress UIs.
- **Binary check** — on finish, `pipeline.sh` drops a single sentinel next to the
  manifest: **`_SUCCESS`** (run completed) xor **`_FAILED`** (any phase failed or
  the run died). Dirt-cheap existence check: `aws s3 ls .../runs/<id>/outputs/_SUCCESS`.
  An EXIT trap guarantees `_FAILED` even on a `set -e` abort or signal.

## Logs

Logs are captured to disk **automatically**, at two granularities:

```
$OUTPUT_BASE/logs/$RUN_ID/<phase>.log          # whole phase (aggregate)
$OUTPUT_BASE/logs/$RUN_ID/<phase>/<step>.log   # one file per step
```

`pipeline.sh` tees each phase to `<phase>.log`; `lib/common.sh::run_step` tees
each step to `<phase>/<step>.log` and records that path in the manifest. Real
exit codes are captured via `PIPESTATUS[0]` (not `tee`'s), so a failing step/
phase still propagates. Synchronous (zero overhead). W&B training logs (sft +
rl only) are async and live separately at `$OUTPUT_BASE/wandb_logs/`.

### Viewing logs

Use `scripts/logs.sh` instead of digging through `outputs/logs/` by hand:

```bash
# Latest run, all phases (concatenated with banners between files)
./scripts/logs.sh

# Just one phase
./scripts/logs.sh --phase graphmert

# One step within a phase
./scripts/logs.sh --phase graphmert --step tokenize

# A specific run (prefix-match: '20260617' → latest run from that day)
./scripts/logs.sh --run 20260617

# Triage view — just the run.failure summary from the manifest
./scripts/logs.sh --error
./scripts/logs.sh -e

# Follow live
./scripts/logs.sh --tail
./scripts/logs.sh -f --phase graphmert

# Inventory of runs
./scripts/logs.sh --list

# Paths only (one per line) — chain with vim, grep, xargs:
./scripts/logs.sh --paths
./scripts/logs.sh --paths --phase graphmert
vim $(./scripts/logs.sh --paths --phase graphmert --step tokenize)
grep -l ERROR $(./scripts/logs.sh --paths)
```

`--error` reads `run_manifest.json` and prints the failed phase, step, exit
code, message, and `log_tail` — usually enough to triage without opening any
log file. Falls back to a "no failure recorded" message on a clean run.

`--list` shows the **current** run with status from the live manifest; older
runs show as `(historical)` because the manifest only tracks the latest run.

### Optional: ship step logs to AWS CloudWatch

Set `CW_LOG_GROUP` to also push each finished step log to CloudWatch Logs
(stream `<run_id>/<phase>/<step>`, recorded as `cw_log_stream` in the manifest).
No-op when unset; non-fatal on failure (local file + S3 stay canonical).
Requires `boto3` + AWS creds. Implemented in `lib/cw_ship.py` (per-step batch
push). For *live* tailing instead of per-step batches, install the CloudWatch
unified agent in `runpod/bootstrap.sh` pointed at `logs/<run_id>/`.

## Output → S3 sync

`scripts/data_prep/sync_outputs.sh` pushes the entire `$OUTPUT_BASE/` to
`s3://${S3_URI}/runs/${RUN_ID}/outputs/`. Excludes `graphrag/cache/*`,
`graphrag/input/*`, `__pycache__/*`, `*.pyc`. Logs ARE included.

`pipeline.sh` calls it:
- After each phase completes (mid-run crash resilience)
- Once more at pipeline end (catch-all)

Both calls are **best-effort** — sync failure prints a warning and continues
(non-fatal). No-op when `S3_URI` is unset (workstation case).

S3 layout produced:

```
s3://<bucket>/dss/runs/<run-id>/outputs/
  ├── graphrag/output/kg_final.csv
  ├── graphmert/final_kg/expanded_kg.parquet
  ├── curriculum_verified/curriculum_verified.json
  ├── sft_checkpoints/...
  ├── rl_checkpoints/...
  ├── logs/<run-id>/extract.log, graphmert.log, ...        # per-phase
  ├── logs/<run-id>/extract/chunk.log, ...                 # per-step
  ├── run_manifest.json
  └── _SUCCESS  (or _FAILED)                                # completion sentinel
```

## Discovering phases and steps

```bash
./scripts/pipeline.sh --list                    # table of all phases + their steps
./scripts/pipeline.sh --list --phase graphmert  # just graphmert's steps
```

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
├── runpod/                 # RunPod-specific orchestration
│   ├── launch.sh           # workstation: POST pod to RunPod API
│   ├── bootstrap.sh        # pod: clone + setup.sh + .env
│   └── serverless_smoke.sh # smoke-test a serverless endpoint
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
