#!/usr/bin/env bash
# sync_outputs.sh — push pipeline outputs to S3 under a per-run prefix.
#
# Two modes:
#   one-shot (default):   sync once and exit. This is what pipeline.sh calls
#                         after each phase boundary.
#   loop (--loop):        sync repeatedly every $INTERVAL seconds until killed.
#                         Use this from a SECOND ssh session when the running
#                         pipeline.sh didn't have S3_SYNC_INTERVAL_SEC set at
#                         startup — same effect as the in-pipeline background
#                         loop, just driven externally.
#
# Env:
#   S3_URI                 required, program root (e.g. s3://enlibra/dss)
#   RUN_ID                 set by pipeline.sh; auto-detected from latest run if absent
#   OUTPUT_BASE            defaults to $REPO_ROOT/outputs
#   AWS_PROFILE            optional
#   S3_SYNC_INTERVAL_SEC   default interval for --loop mode (default 300)
#
# Flags:
#   --loop / -l            background-loop mode (Ctrl-C to stop)
#   --interval N / -i N    override the interval (seconds; min 10)
#   --run-id <id>          override RUN_ID (default: env var, or auto from latest)
#   --help / -h
#
# Examples:
#   ./scripts/data_prep/sync_outputs.sh                        # one-shot, uses env $RUN_ID
#   ./scripts/data_prep/sync_outputs.sh --loop                 # loop every $S3_SYNC_INTERVAL_SEC s
#   ./scripts/data_prep/sync_outputs.sh -l -i 60               # loop every 60 s
#   ./scripts/data_prep/sync_outputs.sh -l --run-id 20260618-095251-pilot-8317524
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

usage() { sed -n '2,/^$/p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --loop|-l)     LOOP=1; shift ;;
        --interval|-i) INTERVAL="$2"; shift 2 ;;
        --run-id)      RUN_ID_OVERRIDE="$2"; shift 2 ;;
        --help|-h)     usage; exit 0 ;;
        *) echo "unknown arg: $1" >&2; usage >&2; exit 1 ;;
    esac
done

# --- env + RUN_ID resolution ------------------------------------------------
: "${S3_URI:?S3_URI must be set (e.g. s3://enlibra/dss); source \$SI_HOME/.env first}"
command -v aws >/dev/null 2>&1 || { echo "aws CLI not found"; exit 1; }

OUTPUT_BASE="${OUTPUT_BASE:-$REPO_ROOT/outputs}"
[[ -d "$OUTPUT_BASE" ]] || { echo "OUTPUT_BASE not found: $OUTPUT_BASE"; exit 1; }

# RUN_ID precedence: --run-id > env > auto-detect newest run dir under logs/
if [[ -n "$RUN_ID_OVERRIDE" ]]; then
    RUN_ID="$RUN_ID_OVERRIDE"
elif [[ -z "${RUN_ID:-}" ]]; then
    RUN_ID=$(find "$OUTPUT_BASE/logs" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' 2>/dev/null \
                 | sort -r | head -1 || true)
    [[ -z "$RUN_ID" ]] && { echo "RUN_ID not set and no run dirs found under $OUTPUT_BASE/logs/"; exit 1; }
    echo "[sync_outputs] auto-detected latest RUN_ID: $RUN_ID"
fi

PROFILE_FLAG=""
[[ -n "${AWS_PROFILE:-}" ]] && PROFILE_FLAG="--profile $AWS_PROFILE"

REMOTE="${S3_URI%/}/runs/${RUN_ID}/outputs"

# --- one shot ---------------------------------------------------------------
do_sync() {
    # shellcheck disable=SC2086
    aws $PROFILE_FLAG s3 sync "$OUTPUT_BASE/" "$REMOTE/" \
        --exclude '*/cache/*' \
        --exclude 'graphrag/input/*' \
        --exclude '*.pyc' \
        --exclude '__pycache__/*' \
        --no-progress
}

if [[ "$LOOP" -eq 0 ]]; then
    echo "[sync_outputs] $OUTPUT_BASE/ -> $REMOTE/"
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

echo "[sync_outputs] loop mode: $OUTPUT_BASE/ -> $REMOTE/  every ${INTERVAL}s  (Ctrl-C to stop)"

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
