# RunPod Quick Reference

## TL;DR — three commands

```bash
# workstation: launch pod
./scripts/runpod/launch.sh --profile pilot                                                            
```

```bash
# runpod: bootstrap
bash <(curl -sH "Authorization: token $GITHUB_TOKEN" \
            -H "Accept: application/vnd.github.v3.raw" \
            "https://api.github.com/repos/$GITHUB_REPO/contents/scripts/runpod/bootstrap.sh?ref=$GITHUB_BRANCH")
./scripts/runpod/remote.sh exec 'nohup ./scripts/pipeline.sh --profile 
```

```bash
# runpod: run pipeline
cd $SI_HOME
nohup ./scripts/pipeline.sh --profile $SI_PROFILE --platform runpod > nohup.out 2>&1 &
```

Detailed sections below.

---

## 1. Launch new pod

```bash
# Smoke (12-16 GB GPU, RTX 3060/A4000-class)
./scripts/runpod/launch.sh --profile smoke

# Pilot (48 GB-class — L40/L40S/A100-40)
./scripts/runpod/launch.sh --profile pilot

# Paper (80 GB — H100 / A100-80)
./scripts/runpod/launch.sh --profile paper

# Override corpus path (default per profile YAML)
CORPUS_PATH=corpus/neuroscience/smoke ./scripts/runpod/launch.sh --profile pilot

# Override GPU class
./scripts/runpod/launch.sh --profile pilot --gpu-type "NVIDIA A100 80GB PCIe"

# Dry-run (show what would launch; don't actually)
./scripts/runpod/launch.sh --profile pilot --dry-run
```

---

## 2. Connect to pod

```bash
# Open interactive SSH (uses direct TCP endpoint when available)
./scripts/runpod/remote.sh ssh

# Run a one-off command (uses direct TCP, NOT ssh.runpod.io proxy)
./scripts/runpod/remote.sh exec 'nvidia-smi'
./scripts/runpod/remote.sh exec 'df -h /workspace'
./scripts/runpod/remote.sh exec './scripts/logs.sh -s -d'

# Re-run bootstrap on existing pod (e.g., after config change)
./scripts/runpod/remote.sh bootstrap

# Re-bootstrap with env overrides (operator overrides survive .env source
# per the pipeline.sh patch)
./scripts/runpod/remote.sh bootstrap CORPUS_PATH=corpus/neuroscience/smoke
./scripts/runpod/remote.sh bootstrap STAGES=graphmert,sft

# Per-phase venv install only (skip full bootstrap)
./scripts/runpod/remote.sh bootstrap STAGES=graphmert
```

---

## 3. Launch pipeline on pod

```bash
# === interactive shell on pod ===
./scripts/runpod/remote.sh ssh

# On pod:
cd /workspace/neuro_SI_pipeline
git pull
nohup ./scripts/pipeline.sh --profile pilot --platform runpod > nohup.out 2>&1 &
./scripts/logs.sh -s -d              # check it started

# === single-phase run (skip earlier already-completed) ===
nohup ./scripts/pipeline.sh --profile pilot --platform runpod --phase graphmert > nohup.out 2>&1 &

# === resume a specific run by RUN_ID ===
export RUN_ID=20260620-030304-pilot-d311fd5
nohup ./scripts/pipeline.sh --profile pilot --platform runpod --phase graphmert > nohup.out 2>&1 &
```

---

## 4. Monitor (live)

```bash
# On the pod (most useful — live status + bars)
./scripts/stats.sh --live --system
# Press q or Ctrl-C to quit

# One-shot summary
./scripts/stats.sh --steps

# Tail logs for a specific phase
./scripts/logs.sh --phase graphmert
./scripts/logs.sh --phase graphmert --step preprocess

# Tail the most recent failure (error block)
./scripts/logs.sh --error

# From workstation, hit the pod summary
./scripts/runpod/remote.sh exec './scripts/stats.sh --steps'
```

---

## 5. Diagnose / debug

```bash
# Health-check (is it broken?)
./scripts/diagnose.sh
./scripts/diagnose.sh --phase extract --step build_kg
./scripts/diagnose.sh --phase graphmert --deep

# Quality analysis (is the output any good?)
./scripts/analysis.sh --phase extract --csv outputs/graphrag/output/kg_final.csv
./scripts/analysis.sh --phase graphmert
./scripts/analysis.sh --json | jq                # machine output

# Inspect grounding stats after preprocess
grep 'Grounding results' outputs/logs/<RUN_ID>/graphmert/preprocess.log

# Reproduce extract LLM call in isolation (LOCAL mode)
.venvs/graphrag/bin/python 1_seed_kg/diagnose_llm_extraction.py
```

---

## 6. Resume a failed run

When `graphmert.preprocess` (or any step) fails and you've patched the bug:

```bash
# On pod:
export RUN_ID=20260620-030304-pilot-d311fd5

# Wipe ONLY the failed step's outputs. Keep upstream artifacts.
rm -rf outputs/graphmert/dataset

# If the failure was deeper (e.g., entity_discovery emitted wrong heads),
# wipe the affected sub-steps too:
rm -rf outputs/graphmert/entity_discovery
rm -rf outputs/graphmert/head_positions
rm -rf outputs/graphmert/llm_relations

# Reset the failed phase in the manifest (status -> pending, clear
# failure block so a new attempt isn't masked as the prior failure).
python3 -c "
import json
m = json.load(open('outputs/run_manifest.json'))
p = next(p for p in m['run']['phases'] if p['name'] == 'graphmert')
p['status'] = 'pending'; p.pop('finished_at', None)
for s in p.get('steps', []):
    if s.get('status') in ('failed', 'pending'):
        s['status'] = 'pending'; s.pop('finished_at', None)
m['run']['status'] = 'running'
m['run'].pop('finished_at', None)
m['run'].pop('failure', None)
json.dump(m, open('outputs/run_manifest.json', 'w'), indent=2)
print('manifest reset')
"

# Re-launch only the failed phase
nohup ./scripts/pipeline.sh --profile pilot --platform runpod --phase graphmert > nohup.out 2>&1 &
./scripts/stats.sh --live --system
```

---

## 7. Sync outputs

```bash
# Pod -> S3 (push, automatic at phase boundaries via pipeline.sh)
# To force a manual push from pod:
./scripts/data_prep/sync_outputs.sh

# Loop mode (continuous push every N seconds from a second ssh session):
./scripts/data_prep/sync_outputs.sh --loop --interval 60

# S3 -> workstation (pull, no helper exists yet; use aws s3 sync directly)
set -a; source .env; set +a
aws s3 sync \
    "${S3_URI%/}/runs/<RUN_ID>/outputs/" \
    "./outputs/" \
    --no-progress

# Minimal pull for local testing of just entity_discovery on RTX 3060:
aws s3 sync "${S3_URI%/}/runs/<RUN_ID>/outputs/graphmert/tokenized_inputs/" \
            "./outputs/graphmert/tokenized_inputs/"
aws s3 sync "${S3_URI%/}/runs/<RUN_ID>/outputs/graphmert/stable_tokenizer/" \
            "./outputs/graphmert/stable_tokenizer/"
# Also pull kg_final.csv if you'll run preprocess (grounding):
aws s3 sync "${S3_URI%/}/runs/<RUN_ID>/outputs/graphrag/output/" \
            "./outputs/graphrag/output/"
```

---

## 8. Kill / cleanup

```bash
# Kill the running pipeline (graceful — sends SIGTERM to pgrp, then SIGKILL)
./scripts/kill_pipeline.sh

# Stop the RunPod pod (preserves volume; stops billing for compute)
# Via RunPod UI: pod card -> Stop button
# Via API: see scripts/runpod/launch.sh for the gh-style call

# Nuke everything for a clean rerun (DANGER — destroys all artifacts)
rm -rf outputs/
unset RUN_ID
```

---

## 9. Local-mode (RTX 3060 etc.) — no pod

```bash
# Single phase, smoke profile, on local GPU
SI_DOMAIN=neuroscience SI_PROFILE=smoke \
    ./scripts/pipeline.sh --profile smoke --platform local --phase graphmert

# Single sub-step (e.g., just entity_discovery to validate a prompt migration)
SI_DOMAIN=neuroscience SI_PROFILE=smoke \
    ./scripts/pipeline.sh --profile smoke --platform local \
    --phase graphmert --step preprocess

# Full pipeline locally on smoke
SI_DOMAIN=neuroscience SI_PROFILE=smoke \
    ./scripts/pipeline.sh --profile smoke --platform local
```

---

## 10. Common gotchas

| Symptom | Fix |
|---|---|
| `CORPUS_PATH` you `export`'d is silently clobbered by `.env` value | Either edit `$SI_HOME/.env`, OR ensure `pipeline.sh` has the operator-overrides snapshot patch |
| `./scripts/logs.sh: line 114: name: No such file or directory` | Old bug — pull latest, the shell-embedded python heredoc had unescaped `"` |
| `Status: failed (...)` but you wiped output dirs | Manifest still says `failed` — see §6 manifest-reset Python snippet |
| `Qwen3-XXB falling back to Transformers implementation` warning | Expected — Qwen3 has no native vLLM impl. Slow but works. For perf use smoke's Qwen2.5-3B |
| `pyyaml NOT FOUND` in a phase's venv | `source .venvs/<phase>/bin/activate && uv pip install pyyaml` |
| Diabetes content appears in neuroscience output | Run `grep -ri 'diabetes' --include='*.py' --include='*.yaml' 2_graphmert/ prompts/ domains/` — flag any hardcoded prompts |
| `Keys mismatch ... source ... and {} (target)` from graphmert.preprocess | `Grounding results: success == 0` — likely entity-vocab mismatch between extract and entity_discovery. Pull preprocess.log and check |
| Stats.sh shows `[K` literal in output | Pull latest — that was a tty-detection bug |
| Pod's `git pull` says `Already up to date` after you pushed | Verify the right branch — `./scripts/runpod/remote.sh exec 'git status -sb; git log --oneline -3'` |

---

## 11. Common env variables

| Var | Purpose | Where set |
|---|---|---|
| `SI_DOMAIN` | which domain config (neuroscience / biomed / physics) | shell or `.env` |
| `SI_PROFILE` | profile (smoke / pilot / paper) | exported by `pipeline.sh --profile X` |
| `SI_PLATFORM` | platform (local / runpod / aws) | exported by `pipeline.sh --platform X` |
| `RUN_ID` | resume a specific run | manual export to resume |
| `CORPUS_PATH` | override input corpus dir | shell export OR `.env` |
| `S3_URI` | S3 root for sync (e.g. `s3://enlibra/dss`) | `.env` |
| `STAGES` | comma-separated venv stages for partial bootstrap | `remote.sh bootstrap STAGES=...` |
| `OUTPUT_BASE` | where outputs/ live (default `$REPO_ROOT/outputs`) | rare override |
| `ENV_FILE` | override `.env` location | rare override |

---

## 12. Branch + commit hygiene

```bash
# Verify what's about to commit
git status -sb
git diff --cached --stat

# Per standing rule: no Co-Authored-By, no personal pronouns in commit body
# See feedback memories: depersonalized-commit-messages, no-coauthor-trailer

# Quick lint-style audit before pushing
grep -rnE '\bdiabetes\b' --include='*.py' --include='*.yaml' \
    2_graphmert/ 3_si_curriculum/ prompts/ domains/ \
    | grep -v __pycache__ | grep -v allowed_off_domain
# (zero hits = clean for neuroscience domain)
```

---

## 13. The one-liner I keep typing

```bash
# Workstation -> get pod status + system bars
./scripts/runpod/remote.sh ssh
# (in pod) ./scripts/stats.sh --live --system

# OR remote summary one-shot
./scripts/runpod/remote.sh exec './scripts/stats.sh --steps'
```
