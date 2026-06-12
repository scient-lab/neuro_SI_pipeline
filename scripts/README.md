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
