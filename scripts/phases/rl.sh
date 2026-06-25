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

PHASE_NAME=rl
STEPS=(setup_reward train_grpo eval_rl)
PHASE_DESC="GRPO reinforcement learning on top of merged SFT checkpoint"
STEP_DESCS=(
    "Build GRPO prompts + reward signals (data_prep.py)"
    "GRPO training on top of merged SFT model (rl_training.py)"
    "(no-op) run 3_si_curriculum/test_models/eval_models.py manually"
)

source_venv si_curriculum

OUTPUT_BASE=$(resolve_output_base)
RL_DATASET_DIR="$OUTPUT_BASE/rl_dataset"
RL_CHECKPOINTS_DIR="$OUTPUT_BASE/rl_checkpoints"
mkdir -p "$RL_DATASET_DIR" "$RL_CHECKPOINTS_DIR"

VERIFIED_CURRICULUM="$OUTPUT_BASE/curriculum_verified/curriculum_verified.json"
SFT_MERGED_MODEL="${SFT_MERGED_MODEL:-}"

# If not set explicitly, find the NEWEST merged SFT model by mtime — not
# alphabetical (tail -1), which would grab a stale checkpoint-epoch-N/
# merged_final_model from a prior run with more epochs. Same fix as sft.sh's
# merge selector (twin).
if [[ -z "$SFT_MERGED_MODEL" ]]; then
    SFT_MERGED_MODEL=$(ls -dt "$OUTPUT_BASE/sft_checkpoints"/checkpoint-*/merged_final_model 2>/dev/null | head -1 || true)
fi

DEEPSPEED_CFG="${DEEPSPEED_CFG:-$REPO_ROOT/3_si_curriculum/RL/deepspeed_config.json}"

# --- Steps ---------------------------------------------------------------
step_setup_reward() {
    # audit bug #10 fix: data_prep.py's "rl" mode no longer chains into
    # preprocess_grpo_dataset by default — it just slices the verified
    # curriculum JSON and saves as a DatasetDict, leaving the
    # `question_and_explanation` column intact. rl_training.py:602 calls
    # preprocess_grpo_dataset once on its input, so the chain runs cleanly.
    # (Previously both data_prep AND rl_training preprocessed, and the
    # second call hit KeyError 'question_and_explanation' because the
    # column was discarded by the first pass.)
    log_info "rl :: setup_reward (data_prep.py — slice only; rl_training preprocesses)"
    INPUT_PATH="$VERIFIED_CURRICULUM" OUTPUT_PATH="$RL_DATASET_DIR" \
        python "$REPO_ROOT/3_si_curriculum/RL/data_prep.py" \
        || { log_error "rl.setup_reward failed"; return 1; }
}

step_train_grpo() {
    log_info "rl :: train_grpo (rl_training.py — HfArgumentParser)"
    if [[ -z "$SFT_MERGED_MODEL" || ! -d "$SFT_MERGED_MODEL" ]]; then
        log_error "rl.train_grpo: no merged SFT model found. Run sft phase first or set SFT_MERGED_MODEL."
        return 1
    fi
    # W&B guard is centralized in pipeline.sh::wandb_autodisable (common.sh) and
    # inherited via WANDB_MODE — no per-phase guard needed (audit bug #17 moved
    # here from this script so sft/rl can't drift out of sync again).
    # audit bug #8 fix: rl_training.py:589 reads config.sft_checkpoint_path,
    # NOT config.model_name. Previously passed --model_name which
    # rl_training silently ignored, then tried from_pretrained("") and
    # crashed. Pass via --sft_checkpoint_path (which the field name expects).
    #
    # audit bug #14 fix: --deepspeed is YAML-gated via rl.use_deepspeed.
    # The default config (3_si_curriculum/RL/deepspeed_config.json) is
    # ZeRO-3 + multi-GPU; on single-GPU smoke/pilot, ZeRO-3's
    # init_distributed() falls through to mpi4py which isn't installed.
    # Single-GPU doesn't benefit from ZeRO-3 partitioning anyway.
    # Paper-grade default in configs/default.yaml::rl.use_deepspeed=true;
    # pilot.yaml and smoke.yaml override to false for single-GPU runs.
    local DS_ARGS=()
    local USE_DS
    USE_DS=$(get_phase_param rl use_deepspeed true)
    if [[ "$USE_DS" == "true" || "$USE_DS" == "True" || "$USE_DS" == "1" ]]; then
        DS_ARGS=(--deepspeed "$DEEPSPEED_CFG")
        log_info "rl.train_grpo: deepspeed ENABLED — config: $DEEPSPEED_CFG"
    else
        log_info "rl.train_grpo: single-GPU mode (rl.use_deepspeed=$USE_DS)"
    fi
    ( cd "$REPO_ROOT/3_si_curriculum/RL" && \
      python rl_training.py \
          --sft_checkpoint_path "$SFT_MERGED_MODEL" \
          --dataset_path        "$RL_DATASET_DIR" \
          --output_dir          "$RL_CHECKPOINTS_DIR" \
          "${DS_ARGS[@]}" \
          --wandb_project       "${WANDB_PROJECT:-${SI_DOMAIN:-neuroscience}_rl_kg}" ) \
        || { log_error "rl.train_grpo failed"; return 1; }
}

step_eval_rl() {
    log_info "rl :: eval_rl (no-op — operator runs 3_si_curriculum/test_models/eval_models.py separately)"
}

# --- Step dispatch -------------------------------------------------------
for step in "${STEPS[@]}"; do
    run_step "$PHASE_NAME" "$step" "step_$step" || exit $?
done
