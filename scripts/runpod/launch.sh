#!/usr/bin/env bash
# scripts/runpod/launch.sh — POST a RunPod pod with env vars pre-injected.
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
#   ./scripts/runpod/launch.sh                              # smoke profile (default)
#   ./scripts/runpod/launch.sh --profile pilot              # bigger GPU
#   ./scripts/runpod/launch.sh --gpu-type "NVIDIA RTX A6000" --disk-gb 200
#   ./scripts/runpod/launch.sh --dry-run                    # print POST body, don't send
#   ./scripts/runpod/launch.sh --discover                   # print discovered GPU chain, don't POST
#
# GPU selection (in order of precedence):
#   1. --gpu-type / RUNPOD_GPU_TYPE       — pin one type, no fallback
#   2. profile vram_gb_min/max + cloud    — dynamic via GET /v1/gpus, VRAM desc (high->low), $/hr tiebreak
#   3. profile gpu_types (list)           — static fallback chain
#   4. profile gpu_type (scalar, legacy)  — single type

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Two levels up: scripts/runpod/launch.sh -> repo root.
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

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
DISCOVER_ONLY=0
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
        --discover)   DISCOVER_ONLY=1;        shift ;;
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

# Read profile.runpod.gpu_types (list) — falls back to the legacy
# scalar `gpu_type` key for back-compat. Emits one GPU-type name per line so
# callers can read into a bash array.
profile_get_gpu_types() {
    uv run --no-project --quiet --with pyyaml python3 -c "
import sys, yaml
with open('$PROFILE_FILE') as f:
    data = yaml.safe_load(f) or {}
runpod = data.get('runpod') or {}
types = runpod.get('gpu_types')
if types is None:
    legacy = runpod.get('gpu_type')
    types = [legacy] if legacy else []
if not types:
    sys.exit(2)
for t in types:
    print(t)
" 2>/dev/null || true
}

# Query RunPod's GET /v1/gpus, filter by VRAM range + cloud-tier
# availability, sort by VRAM descending (high->low), price ascending as
# tiebreak. Emits one GPU-type ID per line.
#
# Args: vram_min  vram_max  cloud_type  ('SECURE' or 'COMMUNITY')
# vram_max=0 means no upper bound. Returns empty (and exits 0) on any failure
# so the caller can fall back to a static chain — we don't want a transient
# RunPod-API hiccup to block a pod launch.
discover_gpus_by_vram() {
    local vmin="$1" vmax="$2" cloud="$3"
    local cloud_field
    case "$cloud" in
        SECURE)    cloud_field="secureCloud" ;;
        COMMUNITY) cloud_field="communityCloud" ;;
        *)         cloud_field="" ;;
    esac

    local resp
    resp=$(curl -sS --max-time 15 \
        -H "Authorization: Bearer ${RUNPOD_API_KEY}" \
        "https://rest.runpod.io/v1/gpus" 2>/dev/null) || return 0

    VMIN="$vmin" VMAX="$vmax" CLOUD_FIELD="$cloud_field" \
    python3 -c "
import json, os, sys
try:
    data = json.loads(os.environ.get('RESP', '') or sys.stdin.read())
except Exception:
    sys.exit(0)

# RunPod returns either a bare list or {'data': [...]} depending on version.
gpus = data if isinstance(data, list) else (data.get('data') or [])
vmin = int(os.environ['VMIN'])
vmax_raw = int(os.environ['VMAX'])
vmax = vmax_raw if vmax_raw > 0 else 10**9
cloud_field = os.environ.get('CLOUD_FIELD') or ''

def vram_gb(g):
    for k in ('memoryInGb', 'memory_in_gb', 'gpuMemoryInGb', 'vramGb', 'memoryGB'):
        v = g.get(k)
        if isinstance(v, (int, float)):
            return int(v)
    return 0

def price(g):
    # Try several common field names; fall back to large value so unknown-priced
    # cards rank last but still appear.
    for k in ('communityPrice', 'securePrice', 'lowestPrice', 'price'):
        v = g.get(k)
        if isinstance(v, dict):
            v = v.get('uninterruptablePrice') or v.get('price') or v.get('amount')
        if isinstance(v, (int, float)):
            return float(v)
    return 999.0

def cloud_ok(g):
    if not cloud_field:
        return True
    val = g.get(cloud_field)
    if isinstance(val, bool):
        return val
    return True  # unknown shape: don't filter out

def gpu_id(g):
    return g.get('id') or g.get('displayName') or g.get('name') or ''

matches = [g for g in gpus if vmin <= vram_gb(g) <= vmax and cloud_ok(g)]
# High-to-low: try the largest-VRAM GPU first; cheapest wins within a VRAM tier.
matches.sort(key=lambda g: (-vram_gb(g), price(g)))
for g in matches:
    gid = gpu_id(g)
    if gid:
        print(gid)
" <<< "$resp" 2>/dev/null || true
}

# Resolve the GPU chain. Precedence:
#   1. CLI --gpu-type / env RUNPOD_GPU_TYPE   → single element (explicit = no fallback)
#   2. profile vram_gb_min                    → dynamic discovery via /v1/gpus
#   3. profile gpu_types (static list)        → fallback if discovery returns empty
#   4. profile gpu_type (legacy scalar)       → single element
[[ -z "${RUNPOD_CLOUD_TYPE:-}" ]] && RUNPOD_CLOUD_TYPE=$(profile_get cloud_type)
[[ -z "${RUNPOD_DISK_GB:-}"    ]] && RUNPOD_DISK_GB=$(profile_get disk_gb)
[[ -z "${RUNPOD_NUM_GPUS:-}"   ]] && RUNPOD_NUM_GPUS=$(profile_get num_gpus)

VRAM_GB_MIN=$(profile_get vram_gb_min)
VRAM_GB_MAX=$(profile_get vram_gb_max)

GPU_TYPE_CHAIN=()
DISCOVERY_USED=0
if [[ -n "${RUNPOD_GPU_TYPE:-}" ]]; then
    GPU_TYPE_CHAIN=("$RUNPOD_GPU_TYPE")
elif [[ -n "$VRAM_GB_MIN" ]]; then
    DISCOVERY_USED=1
    while IFS= read -r line; do
        [[ -n "$line" ]] && GPU_TYPE_CHAIN+=("$line")
    done < <(discover_gpus_by_vram "$VRAM_GB_MIN" "${VRAM_GB_MAX:-0}" "$RUNPOD_CLOUD_TYPE")
    echo "[discover] /v1/gpus filtered by vram=${VRAM_GB_MIN}-${VRAM_GB_MAX:-∞}GB cloud=${RUNPOD_CLOUD_TYPE} → ${#GPU_TYPE_CHAIN[@]} types"
fi

# If no chain yet (no explicit override, no vram range, OR discovery returned nothing),
# fall back to the static gpu_types / gpu_type list from the profile.
if [[ ${#GPU_TYPE_CHAIN[@]} -eq 0 ]]; then
    if [[ "$DISCOVERY_USED" -eq 1 ]]; then
        echo "[discover] empty result, falling back to static gpu_types list"
    fi
    while IFS= read -r line; do
        [[ -n "$line" ]] && GPU_TYPE_CHAIN+=("$line")
    done < <(profile_get_gpu_types)
fi

if [[ "$DISCOVER_ONLY" -eq 1 ]]; then
    echo "[discover] resolved chain (${#GPU_TYPE_CHAIN[@]} types, in attempt order):"
    for t in "${GPU_TYPE_CHAIN[@]}"; do echo "  - $t"; done
    exit 0
fi

if [[ ${#GPU_TYPE_CHAIN[@]} -eq 0 ]]; then
    echo "✗ no GPU types resolved (profile $SI_PROFILE has no runpod.gpu_types / gpu_type and no CLI/env override)" >&2
    exit 1
fi
for var in RUNPOD_CLOUD_TYPE RUNPOD_DISK_GB RUNPOD_NUM_GPUS; do
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
for k in ("GITHUB_TOKEN", "GEMINI_API_KEY", "HF_TOKEN", "WANDB_API_KEY",
          "S3_URI", "CORPUS_PATH", "AWS_ACCESS_KEY_ID",
          "AWS_SECRET_ACCESS_KEY", "AWS_DEFAULT_REGION",
          "S3_SYNC_INTERVAL_SEC", "AWS_CLOUDWATCH_LOG_GROUP",
          # Health monitor (scripts/monitor.sh): health CSVs always on; kill opt-in.
          "MONITOR_INTERVAL", "MONITOR_KILL_ON_FAIL", "MONITOR_MAX_RUNTIME",
          "MONITOR_IDLE_MIN", "MONITOR_DISK_CRIT", "MONITOR_ENABLED",
          # Back-compat grace (superseded monitor_pipeline.sh / monitor.sh --fail-grace)
          "MONITOR_TIMEOUT", "MONITOR_FAIL_GRACE",
          # RunPod control-plane API (required for monitor.sh to kill pod on failure)
          "RUNPOD_API_KEY",
          # Pod-side diagnostics (vllm_smoke.sh, diagnose_llm_extraction.py)
          # point at a separate vLLM serving pod via the OpenAI-compatible API.
          "VLLM_ENDPOINT_URL", "VLLM_API_KEY"):
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
export RUNPOD_CLOUD_TYPE RUNPOD_DISK_GB RUNPOD_NUM_GPUS

if [[ $DRY_RUN -eq 1 ]]; then
    echo "[dry-run] GPU fallback chain (${#GPU_TYPE_CHAIN[@]} types):"
    for t in "${GPU_TYPE_CHAIN[@]}"; do echo "  - $t"; done
    echo
    echo "[dry-run] POST body for first GPU type (secrets masked):"
    RUNPOD_GPU_TYPE="${GPU_TYPE_CHAIN[0]}" export RUNPOD_GPU_TYPE
    build_post_body | python3 -c "
import json, re, sys
body = json.loads(sys.stdin.read())
for k in list(body.get('env', {}).keys()):
    if re.search(r'(TOKEN|KEY|SECRET)', k, re.I) and body['env'].get(k):
        body['env'][k] = '***'
print(json.dumps(body, indent=2))
"
    exit 0
fi

# --- POST loop (try each GPU type until one succeeds) --------------------
# The RunPod API returns 500 + "no instances currently available" when a
# specific GPU type has no capacity. We treat that as retryable on the next
# type in the chain. Any other error (auth, quota, bad payload) aborts.
ATTEMPTS=()
POD_ID=""
LAST_RESP=""

for gpu_type in "${GPU_TYPE_CHAIN[@]}"; do
    export RUNPOD_GPU_TYPE="$gpu_type"
    POD_BODY=$(build_post_body)

    echo "[POST] https://rest.runpod.io/v1/pods (profile=$SI_PROFILE, gpu=$gpu_type, disk=${RUNPOD_DISK_GB}GB)"

    LAST_RESP=$(curl -sS -X POST \
        -H "Authorization: Bearer ${RUNPOD_API_KEY}" \
        -H "Content-Type: application/json" \
        -d "$POD_BODY" \
        https://rest.runpod.io/v1/pods)

    POD_ID=$(echo "$LAST_RESP" | python3 -c "
import json, sys
try:
    print(json.load(sys.stdin).get('id', ''))
except Exception:
    pass
")

    if [[ -n "$POD_ID" ]]; then
        ATTEMPTS+=("$gpu_type: ✓ $POD_ID")
        break
    fi

    # No pod id → parse the error. If it's "no instances available", try next.
    ERROR_MSG=$(echo "$LAST_RESP" | python3 -c "
import json, sys
try:
    print(json.load(sys.stdin).get('error', '') or '')
except Exception:
    pass
")
    ATTEMPTS+=("$gpu_type: ✗ ${ERROR_MSG:-unknown error}")

    if echo "$ERROR_MSG" | grep -qiE 'no instances|not available|out of (stock|capacity)'; then
        echo "  → no capacity for '$gpu_type', trying next…"
        continue
    fi

    # Any other error: stop iterating, surface it.
    echo "✗ non-capacity error from RunPod, aborting fallback chain:" >&2
    echo "$LAST_RESP" | python3 -m json.tool >&2 2>/dev/null || echo "$LAST_RESP" >&2
    echo >&2
    echo "Attempts:" >&2
    for a in "${ATTEMPTS[@]}"; do echo "  - $a" >&2; done
    exit 1
done

if [[ -z "$POD_ID" ]]; then
    echo "✗ no capacity on any GPU type in the chain" >&2
    echo "Attempts:" >&2
    for a in "${ATTEMPTS[@]}"; do echo "  - $a" >&2; done
    echo >&2
    echo "Options:" >&2
    echo "  - rerun later (capacity fluctuates by the minute)" >&2
    echo "  - rerun with --cloud-type SECURE (or COMMUNITY) to switch tiers" >&2
    echo "  - rerun with --gpu-type \"<exact name>\" to pin a specific type" >&2
    exit 1
fi

echo "✓ pod created: $POD_ID (gpu=$RUNPOD_GPU_TYPE)"
echo
echo "Next steps — once SSH is reachable on the pod:"
echo
echo "  # Option A (one-line bootstrap via curl):"
echo "  ssh root@<pod-host> -p <port> 'bash <(curl -sH \"Authorization: token \$GITHUB_TOKEN\" \\"
echo "       -H \"Accept: application/vnd.github.v3.raw\" \\"
echo "       \"https://api.github.com/repos/${GITHUB_REPO}/contents/scripts/runpod/bootstrap.sh?ref=${GITHUB_BRANCH}\")'"
echo
echo "  # Option B (clone first, then run locally):"
echo "  ssh root@<pod-host> -p <port>"
echo "  cd ${SI_HOME} && ./scripts/runpod/bootstrap.sh"
echo
echo "Then run the pipeline:"
echo "  cd ${SI_HOME} && ./scripts/pipeline.sh --profile ${SI_PROFILE} --platform runpod"
