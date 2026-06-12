#!/usr/bin/env bash
# runpod_bootstrap.sh - one-shot bootstrap on a fresh RunPod pod.
#
# Run on the pod (NOT the workstation). Two invocation modes:
#
#   1. Curl pipe (no local checkout yet):
#      bash <(curl -sH "Authorization: token $GITHUB_TOKEN" \
#                  -H "Accept: application/vnd.github.v3.raw" \
#                  "https://api.github.com/repos/$GITHUB_REPO/contents/scripts/runpod_bootstrap.sh?ref=$GITHUB_BRANCH")
#
#   2. Local (after the repo is already cloned):
#      cd $SI_HOME && ./scripts/runpod_bootstrap.sh
#
# What it does:
#   1. preflight: apt install git/curl if missing
#   2. install uv if not present
#   3. clone neuro_SI_pipeline using $GITHUB_TOKEN (idempotent — pulls if already cloned)
#   4. run ./setup.sh to create the 3 uv venvs (.venvs/graphrag/graphmert/si_curriculum)
#   5. write $SI_HOME/.env with the LLM API keys injected at pod-create time
#
# Env vars (all injected by scripts/launch_runpod.sh at pod create time):
#   SI_HOME          /workspace/neuro_SI_pipeline   (where to clone)
#   SI_PROFILE       smoke / pilot / paper          (informational; used by pipeline.sh)
#   GITHUB_TOKEN     PAT with 'repo' scope          (required to clone)
#   GITHUB_REPO      scient-lab/neuro_SI_pipeline
#   GITHUB_BRANCH    dev
#   GEMINI_API_KEY   (required for curriculum phases)
#   HF_TOKEN         (required for gated HF models)
#   WANDB_API_KEY    (optional)
#   STAGES           all | graphrag | graphmert | si_curriculum | csv list
#                    (which venvs to create; default: all)

set -euo pipefail

SI_HOME="${SI_HOME:-/workspace/neuro_SI_pipeline}"
GITHUB_REPO="${GITHUB_REPO:-scient-lab/neuro_SI_pipeline}"
GITHUB_BRANCH="${GITHUB_BRANCH:-dev}"
STAGES="${STAGES:-all}"

echo "=== preflight ==="
echo "  SI_HOME       : $SI_HOME"
echo "  GITHUB_REPO   : $GITHUB_REPO"
echo "  GITHUB_BRANCH : $GITHUB_BRANCH"
echo "  STAGES        : $STAGES"
echo

require() {
    if [[ -z "${!1:-}" ]]; then
        echo "✗ missing env: $1 (inject via .env.runpod on the workstation)" >&2
        return 1
    fi
}
require GITHUB_TOKEN || exit 1

# --- 1. apt install ------------------------------------------------------
need_apt=()
command -v git  >/dev/null 2>&1 || need_apt+=(git)
command -v curl >/dev/null 2>&1 || need_apt+=(curl)
if [[ ${#need_apt[@]} -gt 0 ]]; then
    echo "=== apt install: ${need_apt[*]} ==="
    apt-get update -qq
    apt-get install -y -qq "${need_apt[@]}"
fi

# --- 2. install uv -------------------------------------------------------
if ! command -v uv >/dev/null 2>&1; then
    echo "=== install uv ==="
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
echo "  uv: $(uv --version)"

# --- 3. clone repo (idempotent) ------------------------------------------
if [[ ! -d "$SI_HOME/.git" ]]; then
    echo "=== clone $GITHUB_REPO@$GITHUB_BRANCH -> $SI_HOME ==="
    mkdir -p "$(dirname "$SI_HOME")"
    git clone --branch "$GITHUB_BRANCH" \
        "https://${GITHUB_TOKEN}@github.com/${GITHUB_REPO}.git" "$SI_HOME"
else
    echo "=== repo already cloned, pulling latest ==="
    cd "$SI_HOME"
    git fetch origin "$GITHUB_BRANCH"
    git checkout "$GITHUB_BRANCH"
    git pull --ff-only origin "$GITHUB_BRANCH"
fi

cd "$SI_HOME"

# --- 4. create venvs ------------------------------------------------------
echo "=== ./setup.sh $STAGES ==="
if [[ "$STAGES" == "all" ]]; then
    ./setup.sh
else
    IFS=',' read -ra STAGE_LIST <<< "$STAGES"
    for s in "${STAGE_LIST[@]}"; do
        ./setup.sh "$s"
    done
fi

# --- 5. write .env --------------------------------------------------------
echo "=== write $SI_HOME/.env ==="
ENV_FILE="$SI_HOME/.env"
{
    [[ -n "${GEMINI_API_KEY:-}" ]] && echo "GEMINI_API_KEY=$GEMINI_API_KEY"
    [[ -n "${HF_TOKEN:-}"       ]] && echo "HF_TOKEN=$HF_TOKEN"
    [[ -n "${WANDB_API_KEY:-}"  ]] && echo "WANDB_API_KEY=$WANDB_API_KEY"
} > "$ENV_FILE"
chmod 600 "$ENV_FILE"

# Soft-warn for missing optional secrets — they're not fatal here, but
# the pipeline phases will fail if they need a value that isn't set.
for k in GEMINI_API_KEY HF_TOKEN; do
    if [[ -z "${!k:-}" ]]; then
        echo "  ⚠  $k not set — phases that depend on it will fail"
    fi
done

echo
echo "✓ bootstrap complete"
echo
echo "Run the pipeline:"
echo "  cd $SI_HOME"
echo "  source .env"
echo "  ./scripts/pipeline.sh --profile ${SI_PROFILE:-smoke} --platform runpod"
