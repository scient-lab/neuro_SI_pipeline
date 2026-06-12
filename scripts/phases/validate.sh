#!/usr/bin/env bash
# Phase: validate - two-LLM consensus filter on candidate triples.
# Delegates to 1_seed_kg. Venv: graphrag.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
# shellcheck source=../lib/common.sh
source "$SCRIPT_DIR/../lib/common.sh"
# shellcheck source=../lib/venv.sh
source "$SCRIPT_DIR/../lib/venv.sh"

STEP_FILTER="${1:-all}"
export PIPELINE_STEP_FILTER="$STEP_FILTER"

STEPS=(prepare_candidates llm_check_a llm_check_b consensus dedupe_merge emit_seed_kg)

source_venv graphrag

for step in "${STEPS[@]}"; do
    if step_enabled "$step"; then
        log_info "validate :: $step (stub - wire to 1_seed_kg)"
    fi
done
