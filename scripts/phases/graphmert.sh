#!/usr/bin/env bash
# Phase: graphmert - masked-node-modeling KG expansion.
# Delegates to 2_graphmert. Venv: graphmert.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
# shellcheck source=../lib/common.sh
source "$SCRIPT_DIR/../lib/common.sh"
# shellcheck source=../lib/venv.sh
source "$SCRIPT_DIR/../lib/venv.sh"

STEP_FILTER="${1:-all}"
export PIPELINE_STEP_FILTER="$STEP_FILTER"

STEPS=(tokenize preprocess train_mnm predict_tails validate_predictions expand_kg)

source_venv graphmert

for step in "${STEPS[@]}"; do
    if step_enabled "$step"; then
        log_info "graphmert :: $step (stub - wire to 2_graphmert)"
    fi
done
