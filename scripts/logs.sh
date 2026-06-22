#!/usr/bin/env bash
# logs.sh — view per-phase / per-step logs from a pipeline run.
#
# Logs live at $OUTPUT_BASE/logs/<run_id>/:
#   <phase>.log              <- from pipeline.sh's tee  (phase-level)
#   <phase>/<step>.log       <- from lib/common.sh::run_step (step-level)
#
# Manifest at $OUTPUT_BASE/run_manifest.json gives status + failure info for
# the LATEST run (older runs are superseded; logs persist but status doesn't).
#
# Flag syntax mirrors pipeline.sh (--phase/--step). Usage examples:
#   ./scripts/logs.sh                                    # latest run, all phases
#   ./scripts/logs.sh --phase graphmert                  # one phase
#   ./scripts/logs.sh --phase graphmert --step tokenize  # one step
#   ./scripts/logs.sh --run 20260617 --error             # triage today's failure
#   ./scripts/logs.sh --tail                             # follow live
#   ./scripts/logs.sh --list                             # available runs
#   ./scripts/logs.sh --paths                            # just paths (pipe to vim/grep)
#
# For run-status table (was: logs.sh --summary / --details), use
#   ./scripts/stats.sh --steps --live --system
# logs.sh is intentionally scoped to log-FILE viewing only.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
OUTPUT_BASE="${OUTPUT_BASE:-$REPO_ROOT/outputs}"

RUN_ID=""
PHASE=""
STEP=""
TAIL=0
LIST_ONLY=0
ERROR_ONLY=0
PATHS_ONLY=0

usage() {
    sed -n '2,/^$/p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'
}

# --- Argument parsing -------------------------------------------------------
# Note: --summary / -s / --details / -d are intentionally not accepted here.
# Run-status display (per-phase table, system bars, live refresh) lives in
# scripts/stats.sh. logs.sh is log-FILE viewing only.
while [[ $# -gt 0 ]]; do
    case "$1" in
        --run)         RUN_ID="$2"; shift 2 ;;
        --phase)       PHASE="$2"; shift 2 ;;
        --step)        STEP="$2"; shift 2 ;;
        --tail|-f)     TAIL=1; shift ;;
        --list|-l)     LIST_ONLY=1; shift ;;
        --error|-e)    ERROR_ONLY=1; shift ;;
        --paths|-p)    PATHS_ONLY=1; shift ;;
        --summary|-s|--details|-d)
            echo "logs.sh: --summary / --details moved to scripts/stats.sh" >&2
            echo "  ./scripts/stats.sh --steps --live --system" >&2
            exit 2 ;;
        --help|-h)     usage; exit 0 ;;
        *) echo "unknown arg: $1" >&2; usage >&2; exit 1 ;;
    esac
done

LOGS_BASE="$OUTPUT_BASE/logs"
MANIFEST="$OUTPUT_BASE/run_manifest.json"

# --- --list -----------------------------------------------------------------
if [[ "$LIST_ONLY" -eq 1 ]]; then
    if [[ ! -d "$LOGS_BASE" ]]; then
        echo "(no logs dir: $LOGS_BASE)"; exit 0
    fi
    # The manifest only covers ONE run (the latest). For older runs we just
    # show their dir existence — no status is recoverable without per-run
    # manifests.
    current_run=""
    current_status=""
    if [[ -f "$MANIFEST" ]]; then
        read -r current_run current_status < <(python3 -c "
import json
m = json.load(open('$MANIFEST'))['run']
print(m.get('run_id',''), m.get('status',''))
")
    fi
    printf "%-44s %s\n" "RUN_ID" "STATUS"
    printf "%-44s %s\n" "------" "------"
    for rd in $(find "$LOGS_BASE" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' 2>/dev/null | sort -r); do
        if [[ "$rd" == "$current_run" ]]; then
            printf "%-44s %s\n" "$rd" "$current_status (current)"
        else
            printf "%-44s %s\n" "$rd" "(historical)"
        fi
    done
    exit 0
fi

# --- Resolve RUN_ID (default latest, or prefix match) -----------------------
if [[ -z "$RUN_ID" ]]; then
    RUN_ID=$(find "$LOGS_BASE" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' 2>/dev/null | sort -r | head -1 || true)
    [[ -z "$RUN_ID" ]] && { echo "No runs found in $LOGS_BASE/"; exit 1; }
elif [[ ! -d "$LOGS_BASE/$RUN_ID" ]]; then
    mapfile -t matches < <(find "$LOGS_BASE" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' 2>/dev/null | grep -E "^${RUN_ID}" | sort -r)
    if [[ ${#matches[@]} -eq 0 ]]; then
        echo "No run matching: $RUN_ID" >&2; exit 1
    elif [[ ${#matches[@]} -gt 1 ]]; then
        echo "Multiple runs match '$RUN_ID':" >&2
        printf "  %s\n" "${matches[@]}" >&2
        echo "Specify a more specific prefix." >&2
        exit 1
    fi
    RUN_ID="${matches[0]}"
fi

LOG_DIR="$LOGS_BASE/$RUN_ID"

# --- --error / -e (manifest triage view) ------------------------------------
if [[ "$ERROR_ONLY" -eq 1 ]]; then
    [[ -f "$MANIFEST" ]] || { echo "No manifest at $MANIFEST"; exit 1; }
    python3 -c "
import json, sys
m = json.load(open('$MANIFEST'))['run']
this_run = m.get('run_id', '')
if this_run != '$RUN_ID':
    print(f'Note: manifest is for {this_run}, requested $RUN_ID')
    print('(failure info is only available for the run whose manifest is current)')
    sys.exit(0)
print(f'Run:    {this_run}')
print(f'Status: {m.get(\"status\")}')
print(f'Phases: {len(m.get(\"phases\",[]))}')
f = m.get('failure')
if not f:
    print()
    print('(no run.failure recorded)')
    sys.exit(0)
print()
print('FAILURE:')
print(f'  phase:     {f.get(\"phase\")}')
print(f'  step:      {f.get(\"step\", \"(none)\")}')
print(f'  exit_code: {f.get(\"exit_code\")}')
err = f.get('error') or {}
print(f'  message:   {err.get(\"message\")}')
print()
print('Log tail:')
for ln in err.get('log_tail', []):
    print(f'  {ln}')
"
    exit 0
fi

# --- Build the log-file list per filters -----------------------------------
# Explicit ()  init — `declare -a` alone leaves the var unset on some bash
# versions, which trips `set -u` when we read ${#LOG_FILES[@]} on an empty result.
LOG_FILES=()

if [[ -n "$STEP" ]]; then
    # Step-level: nested at <phase>/<step>.log (preferred), or grep the phase log.
    if [[ -n "$PHASE" ]]; then
        f="$LOG_DIR/$PHASE/$STEP.log"
        [[ -f "$f" ]] && LOG_FILES+=("$f")
    else
        # No phase: scan every phase dir for that step.
        for d in "$LOG_DIR"/*/; do
            [[ -d "$d" ]] || continue
            f="${d}${STEP}.log"
            [[ -f "$f" ]] && LOG_FILES+=("$f")
        done
    fi
elif [[ -n "$PHASE" ]]; then
    # Phase-level: phase.log + every step log under <phase>/.
    [[ -f "$LOG_DIR/$PHASE.log" ]] && LOG_FILES+=("$LOG_DIR/$PHASE.log")
    if [[ -d "$LOG_DIR/$PHASE" ]]; then
        while IFS= read -r sl; do LOG_FILES+=("$sl"); done < <(find "$LOG_DIR/$PHASE" -maxdepth 1 -name '*.log' -type f | sort)
    fi
else
    # Everything in the run.
    while IFS= read -r lf; do LOG_FILES+=("$lf"); done < <(find "$LOG_DIR" -name '*.log' -type f | sort)
fi

if [[ ${#LOG_FILES[@]} -eq 0 ]]; then
    echo "No log files matched in $LOG_DIR" >&2
    [[ -n "$PHASE" || -n "$STEP" ]] && echo "  (filters: phase=${PHASE:-*} step=${STEP:-*})" >&2
    exit 1
fi

# --- Render -----------------------------------------------------------------
# --paths: emit just the absolute paths, one per line. Useful for:
#   vim $(./scripts/logs.sh --paths --phase graphmert)
#   grep -l ERROR $(./scripts/logs.sh --paths)
if [[ "$PATHS_ONLY" -eq 1 ]]; then
    for lf in "${LOG_FILES[@]}"; do echo "$lf"; done
    exit 0
fi

echo "=== Run: $RUN_ID  (${#LOG_FILES[@]} file(s)) ==="
echo
echo "Log files:"
for lf in "${LOG_FILES[@]}"; do
    echo "  ${lf#"$REPO_ROOT"/}"
done

if [[ "$TAIL" -eq 1 ]]; then
    # tail -F handles multiple files with banners between them (built-in headers).
    tail -F "${LOG_FILES[@]}"
else
    for lf in "${LOG_FILES[@]}"; do
        echo
        echo "===== ${lf#"$REPO_ROOT"/} ====="
        cat "$lf"
    done
fi
