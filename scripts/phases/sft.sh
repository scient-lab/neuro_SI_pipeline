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

PHASE_NAME=sft
STEPS=(prepare_data train_lora merge_lora eval_sft)
PHASE_DESC="LoRA supervised fine-tuning on verified curriculum"
STEP_DESCS=(
    "Tokenize curriculum Q&A into SFT dataset format"
    "LoRA fine-tune base model on curriculum (trainer.py)"
    "Merge LoRA adapters into base model weights (merge_lora.py)"
    "(no-op) run 3_si_curriculum/test_models/eval_models.py manually"
)

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

# --- Steps ---------------------------------------------------------------
step_prepare_data() {
    log_info "sft :: prepare_data (data_prep.py)"
    # data_prep.py emits a DatasetDict({"train","test"}) with a `text` column
    # via tokenizer.apply_chat_template; it does NOT tokenize (TRL SFTTrainer
    # handles that inside trainer.py using sft.block_size from YAML). So
    # there's no --max_length flag here. Earlier wiring passed --max_length
    # via SFT_MAX_LENGTH, which crashed argparse on 2026-06-24 because
    # data_prep.py never defined that argument.
    ( cd "$REPO_ROOT/3_si_curriculum/training" && \
      python data_prep.py \
          --input_file  "$VERIFIED_CURRICULUM" \
          --output_path "$SFT_DATASET_DIR" \
          --model_name  "$BASE_MODEL" \
          --cache_dir   "${HF_HOME:-$HOME/.cache/huggingface}" ) \
        || { log_error "sft.prepare_data failed"; return 1; }
}

step_train_lora() {
    log_info "sft :: train_lora (trainer.py — HfArgumentParser)"
    # W&B guard is centralized in pipeline.sh::wandb_autodisable (common.sh) and
    # inherited via WANDB_MODE — no per-phase guard needed.
    local NPROC="${NPROC:-1}"

    # Memory/scale knobs from config (sft.*) with single-GPU-safe FALLBACKS.
    # HF's default per_device_train_batch_size=8 × block_size 4096 × an 8B model
    # OOMs a 48 GB GPU (smoke 2026-06-25). batch=1 + gradient_accumulation keeps
    # an effective batch while fitting; gradient_checkpointing trades compute for
    # activation memory. Profiles override in configs/profiles/<p>.yaml::sft.
    local SFT_BATCH SFT_ACCUM SFT_EPOCHS SFT_CKPT
    # Fallbacks mirror configs/default.yaml::sft (only fire if config is
    # unreadable) so degraded-mode behavior matches the real default.
    SFT_BATCH=$(get_phase_param sft per_device_train_batch_size 1)
    SFT_ACCUM=$(get_phase_param sft gradient_accumulation_steps 8)
    SFT_EPOCHS=$(get_phase_param sft num_train_epochs 3)
    SFT_CKPT=$(get_phase_param sft gradient_checkpointing true)

    # Trainer args built ONCE and shared by both launch paths (no duplicated
    # flag lists to drift out of sync). LoRA params come from TrainingConfig
    # defaults wired to merged config.
    local TRAIN_ARGS=(
        --model_name                  "$BASE_MODEL"
        --train_dataset_path          "$SFT_DATASET_DIR"
        --output_dir                  "$SFT_CHECKPOINTS_DIR"
        --wandb_dir                   "$OUTPUT_BASE/wandb_logs"
        --wandb_project               "${WANDB_PROJECT:-${SI_DOMAIN:-neuroscience}_sft_kg}"
        --per_device_train_batch_size "$SFT_BATCH"
        --gradient_accumulation_steps "$SFT_ACCUM"
        --num_train_epochs            "$SFT_EPOCHS"
        --gradient_checkpointing      "$SFT_CKPT"
    )

    if [[ "$NPROC" == "1" ]]; then
        # NPROC=1: direct python (faster startup, no torchrun rendezvous).
        ( cd "$REPO_ROOT/3_si_curriculum/training" && \
          python trainer.py "${TRAIN_ARGS[@]}" ) \
            || { log_error "sft.train_lora failed"; return 1; }
    else
        ( cd "$REPO_ROOT/3_si_curriculum/training" && \
          torchrun --nproc_per_node="$NPROC" trainer.py "${TRAIN_ARGS[@]}" ) \
            || { log_error "sft.train_lora failed"; return 1; }
    fi
}

step_merge_lora() {
    log_info "sft :: merge_lora (merge_lora.py)"
    local ADAPTER_DIR="${ADAPTER_DIR:-$(ls -d "$SFT_CHECKPOINTS_DIR"/checkpoint-* 2>/dev/null | tail -1)}"
    if [[ -z "$ADAPTER_DIR" || ! -d "$ADAPTER_DIR" ]]; then
        log_error "sft.merge_lora: no checkpoint found in $SFT_CHECKPOINTS_DIR"
        return 1
    fi
    ( cd "$REPO_ROOT/3_si_curriculum/training" && \
      python merge_lora.py \
          --base_model   "$BASE_MODEL" \
          --adapter_path "$ADAPTER_DIR" ) \
        || { log_error "sft.merge_lora failed"; return 1; }
    log_info "Merged SFT model: $ADAPTER_DIR/merged_final_model/"
}

step_eval_sft() {
    log_info "sft :: eval_sft (no-op — operator runs 3_si_curriculum/test_models/eval_models.py separately)"
}

# --- Step dispatch -------------------------------------------------------
for step in "${STEPS[@]}"; do
    run_step "$PHASE_NAME" "$step" "step_$step" || exit $?
done
