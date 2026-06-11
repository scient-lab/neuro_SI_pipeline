#!/bin/bash
#SBATCH --job-name=si-rl
#SBATCH --output=logs/rl_%j.out
#SBATCH --error=logs/rl_%j.err
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --gres=gpu:4
# #SBATCH --mail-type=END,FAIL
# #SBATCH --mail-user=your@email.com

# ============================================================
# RL Training (GRPO)
# Runs reinforcement learning on top of SFT checkpoint.
#
# Required env vars:
#   REPO_DIR        - root of neuro_SI_pipeline checkout
#   OUTPUT_BASE     - scratch output directory
#   MODEL_NAME      - path to SFT-merged checkpoint (base for RL)
#   DATASET_PATH    - path to prepared RL dataset (from RL/data_prep.py)
# Optional:
#   RESUME_CHECKPOINT - path to a prior RL checkpoint to resume from
#   WANDB_DIR         - directory for wandb logs
# ============================================================

set -euo pipefail

: "${REPO_DIR:?REPO_DIR must be set}"
: "${OUTPUT_BASE:?OUTPUT_BASE must be set}"
: "${MODEL_NAME:?MODEL_NAME must be set (path to SFT-merged model)}"
: "${DATASET_PATH:?DATASET_PATH must be set}"

export WANDB_DIR="${WANDB_DIR:-${OUTPUT_BASE}/wandb_logs}"
mkdir -p "${WANDB_DIR}" logs

module purge
# Activate the uv-managed venv (created once — see README "Python environments"):
# source "${REPO_DIR}/.venvs/si_curriculum/bin/activate"

cd "${REPO_DIR}/3_si_curriculum"

python RL/rl_training.py \
    --model_name_or_path "${MODEL_NAME}" \
    --output_dir         "${OUTPUT_BASE}/rl_checkpoints" \
    --dataset_path       "${DATASET_PATH}" \
    ${RESUME_CHECKPOINT:+--resume_from_checkpoint "${RESUME_CHECKPOINT}"}
