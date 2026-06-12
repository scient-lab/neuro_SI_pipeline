#!/usr/bin/env bash
# Phase: rl - GRPO with KG-path-derived reward.
# Delegates to 3_si_curriculum/RL. Venv: si_curriculum.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
# shellcheck source=../lib/common.sh
source "$SCRIPT_DIR/../lib/common.sh"
# shellcheck source=../lib/venv.sh
source "$SCRIPT_DIR/../lib/venv.sh"

STEP_FILTER="${1:-all}"
export PIPELINE_STEP_FILTER="$STEP_FILTER"

STEPS=(setup_reward train_grpo eval_rl)

source_venv si_curriculum

for step in "${STEPS[@]}"; do
    if step_enabled "$step"; then
        log_info "rl :: $step (stub - wire to 3_si_curriculum/RL)"
    fi
done
