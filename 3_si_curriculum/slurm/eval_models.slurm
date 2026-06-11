#!/bin/bash
#SBATCH --job-name=si-eval
#SBATCH --output=logs/eval_%j.out
#SBATCH --error=logs/eval_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:2
# #SBATCH --mail-type=END,FAIL
# #SBATCH --mail-user=your@email.com

# ============================================================
# Model Evaluation
# Runs multi-hop Q&A evaluation across one or more checkpoints.
#
# Required env vars:
#   REPO_DIR        - root of neuro_SI_pipeline checkout
#   EVAL_INPUT_DIR  - directory with eval JSON files
#   EVAL_OUTPUT_DIR - where to write result JSON files
# Optional:
#   MODEL_PATH_1    - path to first model checkpoint
#   MODEL_PATH_2    - path to second model checkpoint
#   MODEL_PATH_3    - path to third model checkpoint
#   BASE_TOKENIZER_PATH - fallback tokenizer if checkpoint lacks one
# ============================================================

set -euo pipefail

: "${REPO_DIR:?REPO_DIR must be set}"
: "${EVAL_INPUT_DIR:?EVAL_INPUT_DIR must be set}"
: "${EVAL_OUTPUT_DIR:?EVAL_OUTPUT_DIR must be set}"

export EVAL_INPUT_DIR EVAL_OUTPUT_DIR
export MODEL_PATH_1="${MODEL_PATH_1:-none}"
export MODEL_PATH_2="${MODEL_PATH_2:-none}"
export MODEL_PATH_3="${MODEL_PATH_3:-none}"
export BASE_TOKENIZER_PATH="${BASE_TOKENIZER_PATH:-}"

mkdir -p "${EVAL_OUTPUT_DIR}" logs

module purge
# Activate the uv-managed venv (created once — see README "Python environments"):
# source "${REPO_DIR}/.venvs/si_curriculum/bin/activate"

cd "${REPO_DIR}/3_si_curriculum"

python test_models/eval_models.py \
    --input_dir  "${EVAL_INPUT_DIR}" \
    --output_dir "${EVAL_OUTPUT_DIR}"
