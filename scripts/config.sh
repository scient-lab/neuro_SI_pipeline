#!/usr/bin/env bash
# scripts/config.sh - effective-config provenance for a run.
#
# Sibling to scripts/stats.sh ("is it running") and scripts/analysis.sh
# ("is the output good"). This answers "what config did each step actually
# use" — read from the per-step config ledger that pipeline_config writes at
# runtime to <run>/config/<phase>.<step>.yaml (post-merge + env overrides,
# with the source layer each value came from). `source: fallback` = no config
# layer set the key, the hard-coded default won.
#
# Usage:
#   ./scripts/config.sh                       # full effective config (all sections)
#   ./scripts/config.sh --models              # MODELS only, nested phase -> step
#   ./scripts/config.sh --params              # phase params only
#   ./scripts/config.sh --prompts             # prompt files used only
#   ./scripts/config.sh --phase curriculum    # one phase
#   ./scripts/config.sh --phase validate --step seed_kg_consensus
#   ./scripts/config.sh --run <prefix>        # a historical run under OUTPUT_BASE
#   ./scripts/config.sh --models --json       # machine output
#   ./scripts/config.sh --tee config.md       # also write to file
#
# Exit: 0 always (provenance view; not a pass/fail gate — use analysis.sh for that).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

ARGS=()
TEE_FILE=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --models|--params|--prompts|--json|--quiet) ARGS+=( "$1" ); shift ;;
        --phase|--step|--run)  ARGS+=( "$1" "$2" ); shift 2 ;;
        --tee)                 TEE_FILE="$2"; shift 2 ;;
        -h|--help)             sed -n '/^#/p' "$0" | sed 's/^# \?//'; exit 0 ;;
        *)                     echo "unknown arg: $1" >&2; exit 1 ;;
    esac
done

# Find a Python with PyYAML (config ledger is YAML). Try venvs, then system.
PY=""
for venv in graphmert graphrag si_curriculum; do
    candidate="$REPO_ROOT/.venvs/$venv/bin/python"
    if [[ -x "$candidate" ]] && "$candidate" -c 'import yaml' 2>/dev/null; then
        PY="$candidate"; break
    fi
done
[[ -z "$PY" ]] && PY="$(command -v python3 || command -v python)"
[[ -z "$PY" ]] && { echo "no python found" >&2; exit 1; }

VIEW="$SCRIPT_DIR/lib/config_view.py"
if [[ -n "$TEE_FILE" ]]; then
    "$PY" "$VIEW" --repo-root "$REPO_ROOT" "${ARGS[@]}" | tee -a "$TEE_FILE"
    exit "${PIPESTATUS[0]}"
fi
exec "$PY" "$VIEW" --repo-root "$REPO_ROOT" "${ARGS[@]}"
