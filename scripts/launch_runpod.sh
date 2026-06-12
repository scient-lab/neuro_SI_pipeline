#!/usr/bin/env bash
# launch_runpod.sh - POST a RunPod pod with env vars pre-injected.
#
# Reads secrets from .env.runpod (workstation-only, gitignored) and hardware
# defaults from configs/profiles/<profile>.yaml::runpod (committed). CLI
# flags win over env vars; env vars win over profile defaults.
#
# Required .env.runpod vars:   RUNPOD_API_KEY
# Strongly recommended:        GITHUB_TOKEN, GEMINI_API_KEY, HF_TOKEN
#                              (otherwise bootstrap on the pod will halt)
#
# Usage:
#   ./scripts/launch_runpod.sh                              # smoke profile (default)
#   ./scripts/launch_runpod.sh --profile pilot              # bigger GPU
#   ./scripts/launch_runpod.sh --gpu-type "NVIDIA RTX A6000" --disk-gb 200
#   ./scripts/launch_runpod.sh --dry-run                    # print POST body, don't send

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# --help is handled before sourcing the env file so it always works.
for arg in "$@"; do
    if [[ "$arg" == "-h" || "$arg" == "--help" ]]; then
        sed -n '2,/^$/p' "${BASH_SOURCE[0]}"
        exit 0
    fi
done

ENV_FILE="${ENV_FILE:-$PROJECT_ROOT/.env.runpod}"

# Capture shell-env overrides BEFORE sourcing .env.runpod (shell wins).
_PRE_SI_PROFILE="${SI_PROFILE:-}"
_PRE_RUNPOD_GPU_TYPE="${RUNPOD_GPU_TYPE:-}"
_PRE_RUNPOD_CLOUD_TYPE="${RUNPOD_CLOUD_TYPE:-}"
_PRE_RUNPOD_DISK_GB="${RUNPOD_DISK_GB:-}"
_PRE_RUNPOD_NUM_GPUS="${RUNPOD_NUM_GPUS:-}"

if [[ -f "$ENV_FILE" ]]; then
    set -a
    # shellcheck disable=SC1091
    source "$ENV_FILE"
    set +a
else
    echo "✗ env file not found: $ENV_FILE" >&2
    echo "  cp .env.runpod.example .env.runpod and fill in" >&2
    exit 1
fi

# Restore shell overrides
[[ -n "$_PRE_SI_PROFILE"        ]] && SI_PROFILE="$_PRE_SI_PROFILE"
[[ -n "$_PRE_RUNPOD_GPU_TYPE"   ]] && RUNPOD_GPU_TYPE="$_PRE_RUNPOD_GPU_TYPE"
[[ -n "$_PRE_RUNPOD_CLOUD_TYPE" ]] && RUNPOD_CLOUD_TYPE="$_PRE_RUNPOD_CLOUD_TYPE"
[[ -n "$_PRE_RUNPOD_DISK_GB"    ]] && RUNPOD_DISK_GB="$_PRE_RUNPOD_DISK_GB"
[[ -n "$_PRE_RUNPOD_NUM_GPUS"   ]] && RUNPOD_NUM_GPUS="$_PRE_RUNPOD_NUM_GPUS"

SI_PROFILE="${SI_PROFILE:-smoke}"
GITHUB_REPO="${GITHUB_REPO:-scient-lab/neuro_SI_pipeline}"
GITHUB_BRANCH="${GITHUB_BRANCH:-dev}"
SI_HOME="${SI_HOME:-/workspace/neuro_SI_pipeline}"
DRY_RUN=0
POD_NAME=""

# Public RunPod image; override via env or --image.
RUNPOD_IMAGE="${RUNPOD_IMAGE:-runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04}"

# --- CLI parsing ---------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --profile)    SI_PROFILE="$2";        shift 2 ;;
        --gpu-type)   RUNPOD_GPU_TYPE="$2";   shift 2 ;;
        --cloud-type) RUNPOD_CLOUD_TYPE="$2"; shift 2 ;;
        --disk-gb)    RUNPOD_DISK_GB="$2";    shift 2 ;;
        --num-gpus)   RUNPOD_NUM_GPUS="$2";   shift 2 ;;
        --image)      RUNPOD_IMAGE="$2";      shift 2 ;;
        --name)       POD_NAME="$2";          shift 2 ;;
        --dry-run)    DRY_RUN=1;              shift ;;
        -h|--help)
            sed -n '2,/^$/p' "$0"
            exit 0
            ;;
        *) echo "unknown arg: $1" >&2; exit 1 ;;
    esac
done

# --- required env check --------------------------------------------------
require() {
    if [[ -z "${!1:-}" ]]; then
        echo "✗ missing env: $1 (see .env.runpod.example)" >&2
        exit 1
    fi
}
require RUNPOD_API_KEY

warn_if_unset() {
    if [[ -z "${!1:-}" ]]; then
        echo "  ⚠  ${1} not set in .env.runpod — bootstrap on the pod will need it"
    fi
}
warn_if_unset GITHUB_TOKEN
warn_if_unset GEMINI_API_KEY
warn_if_unset HF_TOKEN

# --- profile hardware defaults -------------------------------------------
PROFILE_FILE="$PROJECT_ROOT/configs/profiles/${SI_PROFILE}.yaml"
[[ -f "$PROFILE_FILE" ]] || { echo "✗ profile not found: $PROFILE_FILE" >&2; exit 1; }

# Pull profile.runpod.<key> via uv-hosted pyyaml.
profile_get() {
    local key="$1"
    uv run --no-project --quiet --with pyyaml python3 -c "
import sys, yaml
with open('$PROFILE_FILE') as f:
    data = yaml.safe_load(f) or {}
runpod = data.get('runpod') or {}
val = runpod.get('$key')
if val is None:
    sys.exit(2)
print(val)
" 2>/dev/null || true
}

[[ -z "${RUNPOD_GPU_TYPE:-}"   ]] && RUNPOD_GPU_TYPE=$(profile_get gpu_type)
[[ -z "${RUNPOD_CLOUD_TYPE:-}" ]] && RUNPOD_CLOUD_TYPE=$(profile_get cloud_type)
[[ -z "${RUNPOD_DISK_GB:-}"    ]] && RUNPOD_DISK_GB=$(profile_get disk_gb)
[[ -z "${RUNPOD_NUM_GPUS:-}"   ]] && RUNPOD_NUM_GPUS=$(profile_get num_gpus)

for var in RUNPOD_GPU_TYPE RUNPOD_CLOUD_TYPE RUNPOD_DISK_GB RUNPOD_NUM_GPUS; do
    if [[ -z "${!var:-}" ]]; then
        echo "✗ $var not set (profile $SI_PROFILE has no runpod.${var,,} and no CLI/env override)" >&2
        exit 1
    fi
done

POD_NAME="${POD_NAME:-neuro-si-${SI_PROFILE}-$(date -u +%Y%m%d-%H%M)}"

# --- build POST body -----------------------------------------------------
build_post_body() {
    uv run --no-project --quiet --with pyyaml python3 - <<EOF
import json, os
env = {
    "SI_PROFILE":    os.environ["SI_PROFILE"],
    "SI_HOME":       os.environ["SI_HOME"],
    "GITHUB_REPO":   os.environ["GITHUB_REPO"],
    "GITHUB_BRANCH": os.environ["GITHUB_BRANCH"],
}
for k in ("GITHUB_TOKEN", "GEMINI_API_KEY", "HF_TOKEN", "WANDB_API_KEY"):
    v = os.environ.get(k)
    if v:
        env[k] = v

body = {
    "name":              os.environ["POD_NAME"],
    "imageName":         os.environ["RUNPOD_IMAGE"],
    "gpuTypeIds":        [os.environ["RUNPOD_GPU_TYPE"]],
    "gpuCount":          int(os.environ["RUNPOD_NUM_GPUS"]),
    "cloudType":         os.environ["RUNPOD_CLOUD_TYPE"],
    "computeType":       "GPU",
    "containerDiskInGb": int(os.environ["RUNPOD_DISK_GB"]),
    "volumeInGb":        0,
    "ports":             ["22/tcp"],
    "env":               env,
}
print(json.dumps(body, indent=2))
EOF
}

export SI_PROFILE SI_HOME GITHUB_REPO GITHUB_BRANCH POD_NAME RUNPOD_IMAGE
export RUNPOD_GPU_TYPE RUNPOD_CLOUD_TYPE RUNPOD_DISK_GB RUNPOD_NUM_GPUS

POD_BODY=$(build_post_body)

if [[ $DRY_RUN -eq 1 ]]; then
    echo "[dry-run] POST body (secrets masked):"
    echo "$POD_BODY" | python3 -c "
import json, re, sys
body = json.loads(sys.stdin.read())
for k in list(body.get('env', {}).keys()):
    if re.search(r'(TOKEN|KEY|SECRET)', k, re.I) and body['env'].get(k):
        body['env'][k] = '***'
print(json.dumps(body, indent=2))
"
    exit 0
fi

# --- POST ----------------------------------------------------------------
echo "[POST] https://rest.runpod.io/v1/pods (profile=$SI_PROFILE, gpu=$RUNPOD_GPU_TYPE, disk=${RUNPOD_DISK_GB}GB)"

CREATE_RESP=$(curl -sS -X POST \
    -H "Authorization: Bearer ${RUNPOD_API_KEY}" \
    -H "Content-Type: application/json" \
    -d "$POD_BODY" \
    https://rest.runpod.io/v1/pods)

POD_ID=$(echo "$CREATE_RESP" | python3 -c "
import json, sys
try:
    print(json.load(sys.stdin).get('id', ''))
except Exception:
    pass
")

if [[ -z "$POD_ID" ]]; then
    echo "✗ pod creation failed" >&2
    echo "$CREATE_RESP" | python3 -m json.tool >&2 2>/dev/null || echo "$CREATE_RESP" >&2
    exit 1
fi

echo "✓ pod created: $POD_ID"
echo
echo "Next steps — once SSH is reachable on the pod:"
echo
echo "  # Option A (one-line bootstrap via curl):"
echo "  ssh root@<pod-host> -p <port> 'bash <(curl -sH \"Authorization: token \$GITHUB_TOKEN\" \\"
echo "       -H \"Accept: application/vnd.github.v3.raw\" \\"
echo "       \"https://api.github.com/repos/${GITHUB_REPO}/contents/scripts/runpod_bootstrap.sh?ref=${GITHUB_BRANCH}\")'"
echo
echo "  # Option B (clone first, then run locally):"
echo "  ssh root@<pod-host> -p <port>"
echo "  cd ${SI_HOME} && ./scripts/runpod_bootstrap.sh"
echo
echo "Then run the pipeline:"
echo "  cd ${SI_HOME} && ./scripts/pipeline.sh --profile ${SI_PROFILE} --platform runpod"
