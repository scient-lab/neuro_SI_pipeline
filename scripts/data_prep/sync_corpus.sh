#!/usr/bin/env bash
# sync_corpus.sh — sync corpus between S3 and local using a symmetric path.
#
# Env:
#   S3_URI       required, program root (e.g. s3://enlibra/dss)
#   CORPUS_PATH  required, path under both S3 and repo root. Either:
#                  - directory (preferred):  corpus/neuroscience/source_txt
#                  - single .txt file:        corpus/neuroscience/source_txt/purves.txt
#   AWS_PROFILE  optional; passed as --profile to aws
#
# Pulls (default) or pushes:
#   ${S3_URI}/${CORPUS_PATH}  ↔  ${REPO_ROOT}/${CORPUS_PATH}

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

DIRECTION="pull"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --pull) DIRECTION="pull"; shift ;;
        --push) DIRECTION="push"; shift ;;
        -h|--help)
            sed -n '2,/^$/p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "unknown arg: $1" >&2; exit 1 ;;
    esac
done

: "${S3_URI:?S3_URI must be set (e.g. s3://enlibra/dss)}"
: "${CORPUS_PATH:?CORPUS_PATH must be set (e.g. corpus/neuroscience/source_txt)}"
command -v aws >/dev/null 2>&1 || { echo "aws CLI not found"; exit 1; }

PROFILE_FLAG=""
[[ -n "${AWS_PROFILE:-}" ]] && PROFILE_FLAG="--profile $AWS_PROFILE"

# Strip leading/trailing slashes so paths concatenate cleanly.
CP="${CORPUS_PATH#/}"; CP="${CP%/}"
REMOTE="${S3_URI%/}/$CP"
LOCAL="$REPO_ROOT/$CP"

# File-mode if the path ends with .txt; else directory-mode (sync recursively).
if [[ "$CP" == *.txt ]]; then
    mkdir -p "$(dirname "$LOCAL")"
    if [[ "$DIRECTION" == "pull" ]]; then
        echo "pull (file): $REMOTE -> $LOCAL"
        # shellcheck disable=SC2086
        aws $PROFILE_FLAG s3 cp "$REMOTE" "$LOCAL"
    else
        echo "push (file): $LOCAL -> $REMOTE"
        # shellcheck disable=SC2086
        aws $PROFILE_FLAG s3 cp "$LOCAL" "$REMOTE"
    fi
else
    mkdir -p "$LOCAL"
    if [[ "$DIRECTION" == "pull" ]]; then
        echo "pull (dir): $REMOTE/ -> $LOCAL/"
        # shellcheck disable=SC2086
        aws $PROFILE_FLAG s3 sync "$REMOTE/" "$LOCAL/" --exclude ".keep"
    else
        echo "push (dir): $LOCAL/ -> $REMOTE/"
        # shellcheck disable=SC2086
        aws $PROFILE_FLAG s3 sync "$LOCAL/" "$REMOTE/" --exclude ".keep"
    fi
fi
