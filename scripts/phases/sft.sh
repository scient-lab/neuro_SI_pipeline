#!/usr/bin/env bash
# Phase: sft - LoRA supervised fine-tuning.
# Delegates to 3_si_curriculum/training. Venv: si_curriculum.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
# shellcheck source=../lib/common.sh
source "$SCRIPT_DIR/../lib/common.sh"
# shellcheck source=../lib/venv.sh
source "$SCRIPT_DIR/../lib/venv.sh"

STEP_FILTER="${1:-all}"
export PIPELINE_STEP_FILTER="$STEP_FILTER"

STEPS=(prepare_data train_lora merge_lora eval_sft)

source_venv si_curriculum

for step in "${STEPS[@]}"; do
    if step_enabled "$step"; then
        log_info "sft :: $step (stub - wire to 3_si_curriculum/training)"
    fi
done
