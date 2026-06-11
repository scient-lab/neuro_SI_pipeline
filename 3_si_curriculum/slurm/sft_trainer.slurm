#!/bin/bash
#SBATCH --job-name=si-sft
#SBATCH --output=logs/sft_%j.out
#SBATCH --error=logs/sft_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --gres=gpu:4
# #SBATCH --mail-type=END,FAIL
# #SBATCH --mail-user=your@email.com

# ============================================================
# SFT Training (LoRA fine-tuning)
# Fine-tunes base model on verified curriculum Q&A items.
#
# Required env vars:
#   REPO_DIR       - root of neuro_SI_pipeline checkout
#   OUTPUT_BASE    - scratch output directory
#   MODEL_NAME     - HF model id or local path to base model
#   DATASET_PATH   - path to prepared SFT dataset (from data_prep.py)
# Optional:
#   WANDB_DIR      - directory for wandb logs (default: ./wandb_logs)
#   WANDB_PROJECT  - wandb project name
# ============================================================

set -euo pipefail

: "${REPO_DIR:?REPO_DIR must be set}"
: "${OUTPUT_BASE:?OUTPUT_BASE must be set}"
: "${MODEL_NAME:?MODEL_NAME must be set}"
: "${DATASET_PATH:?DATASET_PATH must be set}"

export WANDB_DIR="${WANDB_DIR:-${OUTPUT_BASE}/wandb_logs}"
mkdir -p "${WANDB_DIR}" logs

module purge
# Activate the uv-managed venv (created once — see README "Python environments"):
# source "${REPO_DIR}/.venvs/si_curriculum/bin/activate"

cd "${REPO_DIR}/3_si_curriculum"

torchrun --nproc_per_node=4 training/trainer.py \
    --model_name        "${MODEL_NAME}" \
    --train_dataset_path "${DATASET_PATH}" \
    --output_dir        "${OUTPUT_BASE}/sft_checkpoints" \
    --wandb_dir         "${WANDB_DIR}"
