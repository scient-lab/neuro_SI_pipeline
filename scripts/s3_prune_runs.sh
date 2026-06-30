#!/usr/bin/env bash
# s3_prune_runs.sh — delete old pipeline runs from S3 (${S3_URI}/runs/).
#
# A run's age is read from its RUN_ID name (YYYYMMDD-HHMMSS-profile-hash), NOT
# from S3 LastModified. That's deterministic, matches the documented run-id
# format, and avoids a recursive listing just to find timestamps. A run is
# "old" when its YYYYMMDD date is strictly before (today - N days).
#
# SAFE BY DEFAULT: dry-run. Prints what WOULD be deleted and exits without
# touching S3. Pass --apply to actually delete (with a confirmation prompt
# unless --yes). Runs whose name doesn't parse as a RUN_ID are never deleted.
#
# Env (auto-sourced from $REPO_ROOT/.env):
#   S3_URI               required, program root (e.g. s3://enlibra/dss)
#   AWS_PROFILE          optional -> --profile
#   S3_RETENTION_DAYS    default retention when --days is omitted (default 7)
#
# Flags:
#   --days N             keep runs newer than N days (default: $S3_RETENTION_DAYS or 7)
#   --apply              actually delete (default: dry-run preview only)
#   --yes / -y           skip the confirmation prompt (for cron/non-interactive)
#   --keep-latest        never delete the newest run, even if it is old
#   --profile <p>        AWS profile override
#   --help / -h
#
# Examples:
#   ./scripts/s3_prune_runs.sh                     # dry-run, 7-day retention
#   ./scripts/s3_prune_runs.sh --days 14           # dry-run, 14-day retention
#   ./scripts/s3_prune_runs.sh --days 30 --apply   # delete runs older than 30d
#   ./scripts/s3_prune_runs.sh --apply --yes       # non-interactive (cron)
#
# Exit: 0 ok (incl. nothing to delete), 1 on error.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Auto-source $REPO_ROOT/.env so S3_URI / AWS_* / S3_RETENTION_DAYS reach the
# script when invoked standalone. Idempotent — safe if already sourced.
_env_file="${ENV_FILE:-$REPO_ROOT/.env}"
if [[ -f "$_env_file" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "$_env_file"
    set +a
fi

usage() { sed -n '2,/^$/p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'; }

# --- args -------------------------------------------------------------------
DAYS="${S3_RETENTION_DAYS:-7}"
APPLY=0
ASSUME_YES=0
KEEP_LATEST=0
PROFILE_OVERRIDE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --days)        DAYS="$2"; shift 2 ;;
        --apply)       APPLY=1; shift ;;
        --yes|-y)      ASSUME_YES=1; shift ;;
        --keep-latest) KEEP_LATEST=1; shift ;;
        --profile)     PROFILE_OVERRIDE="$2"; shift 2 ;;
        --help|-h)     usage; exit 0 ;;
        *) echo "unknown arg: $1" >&2; usage >&2; exit 1 ;;
    esac
done

if ! [[ "$DAYS" =~ ^[0-9]+$ ]]; then
    echo "ERROR: --days must be a non-negative integer (got: $DAYS)" >&2
    exit 1
fi

# --- env / prerequisites ----------------------------------------------------
: "${S3_URI:?S3_URI must be set (e.g. s3://enlibra/dss); source \$REPO_ROOT/.env first}"
command -v aws >/dev/null 2>&1 || { echo "aws CLI not found" >&2; exit 1; }
command -v date >/dev/null 2>&1 || { echo "date not found" >&2; exit 1; }

PROFILE_FLAG=""
_profile="${PROFILE_OVERRIDE:-${AWS_PROFILE:-}}"
[[ -n "$_profile" ]] && PROFILE_FLAG="--profile $_profile"

BASE="${S3_URI%/}/runs"

# Cutoff: runs with YYYYMMDD strictly before this are deleted. Day-granular,
# which is sufficient because retention is expressed in whole days.
CUTOFF_YMD="$(date -u -d "$DAYS days ago" +%Y%m%d)"
TODAY_EPOCH="$(date -u +%s)"

human() {
    awk -v b="${1:-0}" 'BEGIN{
        split("B KB MB GB TB PB",u," "); i=1;
        while (b>=1024 && i<6){ b/=1024; i++ }
        if (i==1) printf "%d %s", b, u[i]; else printf "%.1f %s", b, u[i]
    }'
}

# --- list run prefixes ------------------------------------------------------
# `aws s3 ls <prefix>/` emits "PRE <dirname>/" lines for sub-prefixes.
# shellcheck disable=SC2086
mapfile -t ALL_RUNS < <(aws $PROFILE_FLAG s3 ls "${BASE}/" 2>/dev/null \
    | awk '{print $NF}' | tr -d '/' | grep -E '^[0-9]{8}-[0-9]{6}-[a-z0-9-]+$' | sort)

if [[ "${#ALL_RUNS[@]}" -eq 0 ]]; then
    echo "No runs found under ${BASE}/ (or none match the RUN_ID format)."
    exit 0
fi

# Newest run (timestamp prefix sorts chronologically) — protected by --keep-latest.
LATEST="${ALL_RUNS[-1]}"

echo "S3 prune  ${BASE}/"
echo "retention ${DAYS}d  →  delete runs dated before ${CUTOFF_YMD} (UTC)"
[[ "$KEEP_LATEST" -eq 1 ]] && echo "keep-latest: $LATEST is protected"
echo

# --- classify ---------------------------------------------------------------
CANDIDATES=()
TOTAL_BYTES=0
TOTAL_OBJS=0

for rid in "${ALL_RUNS[@]}"; do
    ymd="${rid:0:8}"
    if [[ "$ymd" < "$CUTOFF_YMD" ]]; then
        if [[ "$KEEP_LATEST" -eq 1 && "$rid" == "$LATEST" ]]; then
            continue
        fi
        age_days=$(( (TODAY_EPOCH - $(date -u -d "$ymd" +%s)) / 86400 ))
        # Best-effort size/object count for the preview (one summarize call).
        # shellcheck disable=SC2086
        summ="$(aws $PROFILE_FLAG s3 ls --recursive --summarize "${BASE}/${rid}/" 2>/dev/null || true)"
        objs="$(awk '/Total Objects:/{print $NF}' <<<"$summ")"; objs="${objs:-0}"
        bytes="$(awk '/Total Size:/{print $NF}' <<<"$summ")"; bytes="${bytes:-0}"
        printf '  DELETE  %-34s  %3sd old  %8s  %s obj\n' "$rid" "$age_days" "$(human "$bytes")" "$objs"
        CANDIDATES+=("$rid")
        TOTAL_BYTES=$(( TOTAL_BYTES + bytes ))
        TOTAL_OBJS=$(( TOTAL_OBJS + objs ))
    fi
done

KEPT=$(( ${#ALL_RUNS[@]} - ${#CANDIDATES[@]} ))
echo
echo "summary: ${#CANDIDATES[@]} to delete ($(human "$TOTAL_BYTES"), ${TOTAL_OBJS} objects), ${KEPT} kept of ${#ALL_RUNS[@]} total."

if [[ "${#CANDIDATES[@]}" -eq 0 ]]; then
    echo "Nothing to delete."
    exit 0
fi

# --- dry-run gate -----------------------------------------------------------
if [[ "$APPLY" -eq 0 ]]; then
    echo
    echo "DRY-RUN — no objects deleted. Re-run with --apply to delete the above."
    exit 0
fi

# --- confirmation -----------------------------------------------------------
if [[ "$ASSUME_YES" -eq 0 ]]; then
    if [[ ! -t 0 ]]; then
        echo "ERROR: --apply in a non-interactive shell requires --yes." >&2
        exit 1
    fi
    read -r -p "Permanently delete ${#CANDIDATES[@]} runs ($(human "$TOTAL_BYTES")) from S3? [y/N] " ans
    case "$ans" in
        y|Y|yes|YES) ;;
        *) echo "Aborted."; exit 0 ;;
    esac
fi

# --- delete -----------------------------------------------------------------
rc=0
for rid in "${CANDIDATES[@]}"; do
    echo "[rm] ${BASE}/${rid}/"
    # shellcheck disable=SC2086
    if ! aws $PROFILE_FLAG s3 rm "${BASE}/${rid}/" --recursive --only-show-errors; then
        echo "  WARNING: failed to fully delete $rid" >&2
        rc=1
    fi
done

echo
if [[ "$rc" -eq 0 ]]; then
    echo "Deleted ${#CANDIDATES[@]} runs ($(human "$TOTAL_BYTES") freed)."
else
    echo "Completed with errors; some runs may be partially deleted." >&2
fi
exit "$rc"
