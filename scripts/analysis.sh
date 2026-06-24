#!/usr/bin/env bash
# scripts/analysis.sh - quality analysis for pipeline phase outputs.
#
# Complements scripts/diagnose.sh, which answers "is the run broken".
# This script answers "is the output any good" — relation diversity,
# vocabulary compliance, near-duplicate entities, direction-error
# heuristics, model training curves, prediction coverage, etc.
#
# Each phase has its own analyzer module under scripts/lib/:
#   extract    -> scripts/lib/analysis_extract.py    (kg_final.csv quality)
#   graphmert  -> scripts/lib/analysis_graphmert.py  (preprocess + train +
#                                                     predict + validate)
#   curriculum -> scripts/lib/analysis_curriculum.py (generate + validate +
#                                                     assemble — drop rate,
#                                                     hops, answer balance,
#                                                     traces, diversity)
#
# Usage:
#   ./scripts/analysis.sh                                   # all phases, auto-skip missing
#   ./scripts/analysis.sh --phase extract                   # extract only
#   ./scripts/analysis.sh --phase graphmert                 # graphmert only
#   ./scripts/analysis.sh --phase graphmert --step train_mnm
#   ./scripts/analysis.sh --phase curriculum                # full curriculum analysis
#   ./scripts/analysis.sh --phase curriculum --step generate_qa
#   ./scripts/analysis.sh --phase curriculum --step validate_qa
#   ./scripts/analysis.sh --phase curriculum --step assemble_curriculum
#   ./scripts/analysis.sh --csv logs/kg_final_1.csv         # override extract CSV
#   ./scripts/analysis.sh --json                            # machine output
#   ./scripts/analysis.sh --top 20                          # top-K relations/heads
#   ./scripts/analysis.sh --quiet                           # WARN/FAIL only
#   ./scripts/analysis.sh --run <prefix>                    # historical run
#   ./scripts/analysis.sh --tee report.md                   # also write to file
#
# Exit: 0 clean, 1 on FAILs, 2 on WARN-only (composable with CI).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

PHASE_FILTER=""
STEP_FILTER=""
CSV_OVERRIDE=""
TOP_K=10
JSON_MODE=0
QUIET=0
RUN_PREFIX=""
TEE_FILE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --phase)   PHASE_FILTER="$2"; shift 2 ;;
        --step)    STEP_FILTER="$2"; shift 2 ;;
        --csv)     CSV_OVERRIDE="$2"; shift 2 ;;
        --top)     TOP_K="$2"; shift 2 ;;
        --json)    JSON_MODE=1; shift ;;
        --quiet)   QUIET=1; shift ;;
        --run)     RUN_PREFIX="$2"; shift 2 ;;
        --tee)     TEE_FILE="$2"; shift 2 ;;
        -h|--help) sed -n '/^#/p' "$0" | sed 's/^# \?//'; exit 0 ;;
        *)         echo "unknown arg: $1" >&2; exit 1 ;;
    esac
done

# Find a Python with PyYAML — needed for domain config parsing. Try venvs
# in order, fall back to system python3.
PY=""
for venv in graphmert graphrag si_curriculum; do
    candidate="$REPO_ROOT/.venvs/$venv/bin/python"
    if [[ -x "$candidate" ]] && "$candidate" -c 'import yaml' 2>/dev/null; then
        PY="$candidate"
        break
    fi
done
[[ -z "$PY" ]] && PY="$(command -v python3 || command -v python)"
[[ -z "$PY" ]] && { echo "no python found" >&2; exit 1; }

LIB_DIR="$SCRIPT_DIR/lib"
EXIT_CODE=0

run_phase() {
    local phase="$1"
    local module="$LIB_DIR/analysis_${phase}.py"
    [[ -f "$module" ]] || { echo "no analyzer for phase '$phase' at $module" >&2; return 1; }

    local args=( --repo-root "$REPO_ROOT" --top "$TOP_K" )
    [[ -n "$CSV_OVERRIDE" && "$phase" == "extract" ]] && args+=( --csv "$CSV_OVERRIDE" )
    [[ -n "$STEP_FILTER" ]] && args+=( --step "$STEP_FILTER" )
    [[ -n "$RUN_PREFIX" ]]  && args+=( --run  "$RUN_PREFIX" )
    [[ "$JSON_MODE" -eq 1 ]] && args+=( --json )
    [[ "$QUIET"     -eq 1 ]] && args+=( --quiet )

    if [[ -n "$TEE_FILE" ]]; then
        "$PY" "$module" "${args[@]}" | tee -a "$TEE_FILE"
    else
        "$PY" "$module" "${args[@]}"
    fi
    local rc=${PIPESTATUS[0]}
    [[ "$rc" -gt "$EXIT_CODE" ]] && EXIT_CODE="$rc"
}

if [[ -z "$PHASE_FILTER" || "$PHASE_FILTER" == "extract" ]]; then
    run_phase extract
fi
if [[ -z "$PHASE_FILTER" || "$PHASE_FILTER" == "graphmert" ]]; then
    run_phase graphmert
fi
if [[ -z "$PHASE_FILTER" || "$PHASE_FILTER" == "curriculum" ]]; then
    run_phase curriculum
fi

exit "$EXIT_CODE"
