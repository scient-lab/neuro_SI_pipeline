#!/usr/bin/env bash
# Phase: rl - GRPO with KG-path-derived reward.
# Delegates to 3_si_curriculum/RL. Venv: si_curriculum.
#
# Maps our STEPS onto the Princeton README:
#   setup_reward   data_prep.py (env-var-driven) — prepare the RL dataset
#   train_grpo     rl_training.py — GRPO loop (uses TrainingConfig wired to merged config)
#   eval_rl        (no-op — operator runs test_models/eval_models.py separately)
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

OUTPUT_BASE=$(resolve_output_base)
RL_DATASET_DIR="$OUTPUT_BASE/rl_dataset"
RL_CHECKPOINTS_DIR="$OUTPUT_BASE/rl_checkpoints"
mkdir -p "$RL_DATASET_DIR" "$RL_CHECKPOINTS_DIR"

VERIFIED_CURRICULUM="$OUTPUT_BASE/curriculum_verified/curriculum_verified.json"
SFT_MERGED_MODEL="${SFT_MERGED_MODEL:-}"

# If not set explicitly, find the last merged SFT model.
if [[ -z "$SFT_MERGED_MODEL" ]]; then
    SFT_MERGED_MODEL=$(ls -d "$OUTPUT_BASE/sft_checkpoints"/checkpoint-*/merged_final_model 2>/dev/null | tail -1 || true)
fi

DEEPSPEED_CFG="${DEEPSPEED_CFG:-$REPO_ROOT/3_si_curriculum/RL/deepspeed_config.json}"

for step in "${STEPS[@]}"; do
    if ! step_enabled "$step"; then continue; fi

    case "$step" in
        setup_reward)
            log_info "rl :: setup_reward (data_prep.py — env-var driven)"
            INPUT_PATH="$VERIFIED_CURRICULUM" OUTPUT_PATH="$RL_DATASET_DIR" \
                python "$REPO_ROOT/3_si_curriculum/RL/data_prep.py" \
                || { log_error "rl.setup_reward failed"; exit 1; }
            ;;

        train_grpo)
            log_info "rl :: train_grpo (rl_training.py — HfArgumentParser)"
            if [[ -z "$SFT_MERGED_MODEL" || ! -d "$SFT_MERGED_MODEL" ]]; then
                log_error "rl.train_grpo: no merged SFT model found. Run sft phase first or set SFT_MERGED_MODEL."
                exit 1
            fi
            ( cd "$REPO_ROOT/3_si_curriculum/RL" && \
              python rl_training.py \
                  --model_name   "$SFT_MERGED_MODEL" \
                  --dataset_path "$RL_DATASET_DIR" \
                  --output_dir   "$RL_CHECKPOINTS_DIR" \
                  --deepspeed    "$DEEPSPEED_CFG" \
                  --wandb_project "${WANDB_PROJECT:-rl_neuro_kg}" ) \
                || { log_error "rl.train_grpo failed"; exit 1; }
            ;;

        eval_rl)
            log_info "rl :: eval_rl (no-op — operator runs 3_si_curriculum/test_models/eval_models.py separately)"
            ;;
    esac
done
