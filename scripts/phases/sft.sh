#!/usr/bin/env bash
# Phase: sft - LoRA supervised fine-tuning.
# Delegates to 3_si_curriculum/training. Venv: si_curriculum.
#
# Maps our STEPS onto the Princeton README:
#   prepare_data   data_prep.py — tokenize curriculum into SFT dataset
#   train_lora     trainer.py — LoRA SFT (uses TrainingConfig wired to merged config)
#   merge_lora     merge_lora.py — fold adapter into base model
#   eval_sft       (no-op for now; downstream eval lives in 3_si_curriculum/test_models)
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

OUTPUT_BASE=$(resolve_output_base)
SFT_DATASET_DIR="$OUTPUT_BASE/sft_dataset"
SFT_CHECKPOINTS_DIR="$OUTPUT_BASE/sft_checkpoints"
mkdir -p "$SFT_DATASET_DIR" "$SFT_CHECKPOINTS_DIR"

BASE_MODEL=$(get_model_id base_sft "")
if [[ -z "$BASE_MODEL" ]]; then
    log_error "sft needs models.base_sft in configs/default.yaml"
    exit 1
fi

VERIFIED_CURRICULUM="$OUTPUT_BASE/curriculum_verified/curriculum_verified.json"
SFT_MAX_LENGTH=$(get_phase_param sft block_size 32768)

for step in "${STEPS[@]}"; do
    if ! step_enabled "$step"; then continue; fi

    case "$step" in
        prepare_data)
            log_info "sft :: prepare_data (data_prep.py)"
            ( cd "$REPO_ROOT/3_si_curriculum/training" && \
              python data_prep.py \
                  --input_file  "$VERIFIED_CURRICULUM" \
                  --output_path "$SFT_DATASET_DIR" \
                  --model_name  "$BASE_MODEL" \
                  --max_length  "$SFT_MAX_LENGTH" \
                  --cache_dir   "${HF_HOME:-$HOME/.cache/huggingface}" ) \
                || { log_error "sft.prepare_data failed"; exit 1; }
            ;;

        train_lora)
            log_info "sft :: train_lora (trainer.py — HfArgumentParser)"
            # Single-GPU or multi-GPU via torchrun. Operator can set NPROC.
            NPROC="${NPROC:-1}"
            CMD=(
                "torchrun" "--nproc_per_node=$NPROC"
                "$REPO_ROOT/3_si_curriculum/training/trainer.py"
                "--model_name"          "$BASE_MODEL"
                "--train_dataset_path"  "$SFT_DATASET_DIR"
                "--output_dir"          "$SFT_CHECKPOINTS_DIR"
                "--wandb_dir"           "$OUTPUT_BASE/wandb_logs"
                "--wandb_project"       "${WANDB_PROJECT:-${SI_DOMAIN:-neuroscience}_sft_kg}"
                # LoRA params come from TrainingConfig defaults wired to merged config.
            )
            if [[ "$NPROC" == "1" ]]; then
                # When NPROC=1, prefer direct python invocation (faster startup).
                ( cd "$REPO_ROOT/3_si_curriculum/training" && \
                  python trainer.py \
                      --model_name         "$BASE_MODEL" \
                      --train_dataset_path "$SFT_DATASET_DIR" \
                      --output_dir         "$SFT_CHECKPOINTS_DIR" \
                      --wandb_dir          "$OUTPUT_BASE/wandb_logs" \
                      --wandb_project      "${WANDB_PROJECT:-${SI_DOMAIN:-neuroscience}_sft_kg}" ) \
                    || { log_error "sft.train_lora failed"; exit 1; }
            else
                "${CMD[@]}" || { log_error "sft.train_lora failed"; exit 1; }
            fi
            ;;

        merge_lora)
            log_info "sft :: merge_lora (merge_lora.py)"
            ADAPTER_DIR="${ADAPTER_DIR:-$(ls -d "$SFT_CHECKPOINTS_DIR"/checkpoint-* 2>/dev/null | tail -1)}"
            if [[ -z "$ADAPTER_DIR" || ! -d "$ADAPTER_DIR" ]]; then
                log_error "sft.merge_lora: no checkpoint found in $SFT_CHECKPOINTS_DIR"
                exit 1
            fi
            ( cd "$REPO_ROOT/3_si_curriculum/training" && \
              python merge_lora.py \
                  --base_model   "$BASE_MODEL" \
                  --adapter_path "$ADAPTER_DIR" ) \
                || { log_error "sft.merge_lora failed"; exit 1; }
            log_info "Merged SFT model: $ADAPTER_DIR/merged_final_model/"
            ;;

        eval_sft)
            log_info "sft :: eval_sft (no-op — operator runs 3_si_curriculum/test_models/eval_models.py separately)"
            ;;
    esac
done
