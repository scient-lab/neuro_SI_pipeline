#!/usr/bin/env bash
# sync_outputs.sh — sync pipeline outputs between local and S3.
#
# Direction (--mode):
#   push (default):  local → S3.  What pipeline.sh calls after each phase.
#   pull:            S3 → local.  Use when resuming an existing run on a
#                    fresh pod, or pulling artifacts to a workstation for
#                    analysis.
#
# Cadence:
#   one-shot (default):   sync once and exit
#   loop (--loop):        sync repeatedly every $INTERVAL seconds until killed
#                         (works in both push and pull mode)
#
# Env:
#   S3_URI                 required, program root (e.g. s3://enlibra/dss)
#   RUN_ID                 set by pipeline.sh; auto-detected from latest run if absent
#   OUTPUT_BASE            defaults to $REPO_ROOT/outputs
#   AWS_PROFILE            optional
#   S3_SYNC_INTERVAL_SEC   default interval for --loop mode (default 300)
#
# Flags:
#   --mode push|pull       direction; default push (back-compat with prior behavior)
#   --loop / -l            background-loop mode (Ctrl-C to stop)
#   --interval N / -i N    override the interval (seconds; min 10)
#   --run-id <id>          override RUN_ID (default: env var, or auto from latest)
#   --help / -h
#
# Examples (push — original behaviour):
#   ./scripts/data_prep/sync_outputs.sh                          # one-shot push
#   ./scripts/data_prep/sync_outputs.sh --loop                   # loop push
#   ./scripts/data_prep/sync_outputs.sh -l -i 60                 # loop push every 60s
#   ./scripts/data_prep/sync_outputs.sh -l --run-id 20260618-095251-pilot-8317524
#
# Examples (pull — NEW):
#   ./scripts/data_prep/sync_outputs.sh --mode pull              # latest run on S3 → local
#   ./scripts/data_prep/sync_outputs.sh --mode pull \
#       --run-id 20260622-045429-pilot-3ea615e                   # specific run from S3
#
# S3 layout:
#   ${S3_URI}/runs/${RUN_ID}/outputs/...

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Auto-source $REPO_ROOT/.env so S3_URI / AWS_* / S3_SYNC_INTERVAL_SEC reach
# the script when invoked standalone (e.g. from a second ssh session).
# `set -a` exports each var as we source. Idempotent — safe if the caller
# already sourced .env.
_env_file="${ENV_FILE:-$REPO_ROOT/.env}"
if [[ -f "$_env_file" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "$_env_file"
    set +a
fi

# --- args -------------------------------------------------------------------
LOOP=0
INTERVAL=""
RUN_ID_OVERRIDE=""
MODE="push"   # default preserves prior behavior; --mode pull reverses direction

usage() { sed -n '2,/^$/p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode)        MODE="$2"; shift 2 ;;
        --loop|-l)     LOOP=1; shift ;;
        --interval|-i) INTERVAL="$2"; shift 2 ;;
        --run-id)      RUN_ID_OVERRIDE="$2"; shift 2 ;;
        --help|-h)     usage; exit 0 ;;
        *) echo "unknown arg: $1" >&2; usage >&2; exit 1 ;;
    esac
done

# Validate --mode
case "$MODE" in
    push|pull) ;;
    *) echo "ERROR: --mode must be 'push' or 'pull' (got: $MODE)" >&2; exit 1 ;;
esac

# --- env + RUN_ID resolution ------------------------------------------------
: "${S3_URI:?S3_URI must be set (e.g. s3://enlibra/dss); source \$SI_HOME/.env first}"
command -v aws >/dev/null 2>&1 || { echo "aws CLI not found"; exit 1; }

OUTPUT_BASE="${OUTPUT_BASE:-$REPO_ROOT/outputs}"
# For push: OUTPUT_BASE must exist (we're uploading from it).
# For pull: we'll create it on demand below.
if [[ "$MODE" == "push" && ! -d "$OUTPUT_BASE" ]]; then
    echo "OUTPUT_BASE not found: $OUTPUT_BASE" >&2
    exit 1
fi

PROFILE_FLAG=""
[[ -n "${AWS_PROFILE:-}" ]] && PROFILE_FLAG="--profile $AWS_PROFILE"

# RUN_ID precedence: --run-id > env > auto-detect.
# Auto-detect source differs by mode:
#   push: newest local run dir under $OUTPUT_BASE/logs/
#   pull: newest run prefix in S3 (allows pulling on a fresh pod with no local runs)
if [[ -n "$RUN_ID_OVERRIDE" ]]; then
    RUN_ID="$RUN_ID_OVERRIDE"
elif [[ -z "${RUN_ID:-}" ]]; then
    if [[ "$MODE" == "push" ]]; then
        RUN_ID=$(find "$OUTPUT_BASE/logs" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' 2>/dev/null \
                     | sort -r | head -1 || true)
        [[ -z "$RUN_ID" ]] && {
            echo "RUN_ID not set and no run dirs found under $OUTPUT_BASE/logs/" >&2
            exit 1
        }
    else  # pull
        # shellcheck disable=SC2086
        # AWS S3 ls output for directories: "                           PRE dirname/"
        # awk '{print $NF}' grabs the last field (dirname/), tr -d '/' removes trailing slash
        RUN_ID=$(aws $PROFILE_FLAG s3 ls "${S3_URI%/}/runs/" 2>/dev/null \
                     | awk '{print $NF}' | tr -d '/' \
                     | grep -E '^[0-9]{8}-[0-9]{6}' | sort -r | head -1 || true)
        [[ -z "$RUN_ID" ]] && {
            echo "RUN_ID not set and no runs found in ${S3_URI%/}/runs/" >&2
            exit 1
        }
    fi
    echo "[sync_outputs] auto-detected latest RUN_ID: $RUN_ID  (mode=$MODE)"
fi

# Sanity check: RUN_ID should match the full format (timestamp-timestamp-profile-hash)
if ! [[ "$RUN_ID" =~ ^[0-9]{8}-[0-9]{6}-[a-z0-9-]+$ ]]; then
    echo "WARNING: RUN_ID '$RUN_ID' doesn't match expected format (YYYYMMDD-HHMMSS-profile-hash)" >&2
    echo "  This could indicate a detection bug. Set RUN_ID explicitly with --run-id to override." >&2
fi

REMOTE="${S3_URI%/}/runs/${RUN_ID}/outputs"
LOCAL="$OUTPUT_BASE"
# In pull mode, materialize the local target dir if missing.
[[ "$MODE" == "pull" ]] && mkdir -p "$LOCAL"

# --- one shot ---------------------------------------------------------------
# Source / target depend on --mode. Excludes preserve bandwidth on either
# direction (caches, pycache, raw input corpus that lives in graphrag/input/).
do_sync() {
    local src dst
    if [[ "$MODE" == "push" ]]; then
        src="$LOCAL/"; dst="$REMOTE/"
    else  # pull
        src="$REMOTE/"; dst="$LOCAL/"
    fi
    # shellcheck disable=SC2086
    aws $PROFILE_FLAG s3 sync "$src" "$dst" \
        --exclude '*/cache/*' \
        --exclude 'graphrag/input/*' \
        --exclude '*.pyc' \
        --exclude '__pycache__/*' \
        --no-progress
}

if [[ "$LOOP" -eq 0 ]]; then
    if [[ "$MODE" == "push" ]]; then
        echo "[sync_outputs] push: $LOCAL/ -> $REMOTE/"
    else
        echo "[sync_outputs] pull: $REMOTE/ -> $LOCAL/"
    fi
    do_sync
    exit 0
fi

# --- loop mode --------------------------------------------------------------
# Resolve interval: --interval > $S3_SYNC_INTERVAL_SEC > 300
INTERVAL="${INTERVAL:-${S3_SYNC_INTERVAL_SEC:-300}}"
if ! [[ "$INTERVAL" =~ ^[0-9]+$ ]] || [[ "$INTERVAL" -lt 10 ]]; then
    echo "interval must be a positive integer >= 10 seconds (got: $INTERVAL)" >&2
    exit 1
fi

if [[ "$MODE" == "push" ]]; then
    echo "[sync_outputs] loop push: $LOCAL/ -> $REMOTE/  every ${INTERVAL}s  (Ctrl-C to stop)"
else
    echo "[sync_outputs] loop pull: $REMOTE/ -> $LOCAL/  every ${INTERVAL}s  (Ctrl-C to stop)"
fi

# Trap so Ctrl-C exits cleanly without "Error: aws sync interrupted" noise.
_stop=0
trap '_stop=1; echo; echo "[sync_outputs] stopping after current sync…"' INT TERM

while [[ "$_stop" -eq 0 ]]; do
    ts="$(date -u +'%H:%M:%S')"
    if do_sync >/dev/null 2>&1; then
        echo "[$ts] sync ok"
    else
        echo "[$ts] sync FAILED (non-fatal; will retry)"
    fi
    # Sleep in 1-second slices so SIGINT is responsive.
    for ((i = 0; i < INTERVAL; i++)); do
        [[ "$_stop" -eq 1 ]] && break
        sleep 1
    done
done

echo "[sync_outputs] stopped."
