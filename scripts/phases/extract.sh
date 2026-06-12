#!/usr/bin/env bash
# Phase: extract - single-LLM extraction with closed vocabulary.
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

STEPS=(parse_pdf chunk extract_triples normalize cache)

source_venv graphrag

for step in "${STEPS[@]}"; do
    if step_enabled "$step"; then
        log_info "extract :: $step (stub - wire to 1_seed_kg)"
        # TODO: dispatch to actual step here, e.g.:
        #   ( cd "$REPO_ROOT/1_seed_kg" && python graphrag_index.py --step "$step" )
    fi
done
