#!/usr/bin/env bash
# sync_outputs.sh — push pipeline outputs to S3 under a per-run prefix.
#
# Env:
#   S3_URI       required, program root (e.g. s3://enlibra/dss)
#   RUN_ID       required, set by pipeline.sh at run start
#   OUTPUT_BASE  defaults to $REPO_ROOT/outputs
#   AWS_PROFILE  optional
#
# Layout:
#   ${S3_URI}/runs/${RUN_ID}/outputs/...
#
# Excludes recomputable intermediates (graphrag cache, staged corpus copies)
# to keep the upload small. Logs ARE included — that's the whole point.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

: "${S3_URI:?S3_URI must be set (e.g. s3://enlibra/dss)}"
: "${RUN_ID:?RUN_ID must be set (pipeline.sh writes it at run start)}"
command -v aws >/dev/null 2>&1 || { echo "aws CLI not found"; exit 1; }

OUTPUT_BASE="${OUTPUT_BASE:-$REPO_ROOT/outputs}"
[[ -d "$OUTPUT_BASE" ]] || { echo "OUTPUT_BASE not found: $OUTPUT_BASE"; exit 1; }

PROFILE_FLAG=""
[[ -n "${AWS_PROFILE:-}" ]] && PROFILE_FLAG="--profile $AWS_PROFILE"

REMOTE="${S3_URI%/}/runs/${RUN_ID}/outputs"
echo "[sync_outputs] $OUTPUT_BASE/ -> $REMOTE/"

# shellcheck disable=SC2086
aws $PROFILE_FLAG s3 sync "$OUTPUT_BASE/" "$REMOTE/" \
    --exclude '*/cache/*' \
    --exclude 'graphrag/input/*' \
    --exclude '*.pyc' \
    --exclude '__pycache__/*' \
    --no-progress
