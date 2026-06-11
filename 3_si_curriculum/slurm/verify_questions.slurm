#!/bin/bash
#SBATCH --job-name=si-verify
#SBATCH --output=logs/verify_%j.out
#SBATCH --error=logs/verify_%j.err
#SBATCH --time=6:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
# #SBATCH --mail-type=END,FAIL
# #SBATCH --mail-user=your@email.com

# ============================================================
# Two-LLM Q&A Verification
# Filters curriculum items kept only when both LLMs agree.
#
# Required env vars:
#   REPO_DIR      - root of neuro_SI_pipeline checkout
#   OUTPUT_BASE   - scratch output directory
# Optional:
#   MODEL_IDS     - space-separated pair, e.g. "Qwen/Qwen3-14B meta-llama/Llama-3.1-8B"
#                   defaults to Qwen3-14B + Llama-3.1-8B
# ============================================================

set -euo pipefail

: "${REPO_DIR:?REPO_DIR must be set}"
: "${OUTPUT_BASE:?OUTPUT_BASE must be set}"

CURRICULUM_OUTPUT="${OUTPUT_BASE}/curriculum"
VERIFIED_OUTPUT="${OUTPUT_BASE}/curriculum_verified"
mkdir -p "${VERIFIED_OUTPUT}" logs

MODEL_IDS="${MODEL_IDS:-Qwen/Qwen3-14B meta-llama/Llama-3.1-8B-Instruct}"

module purge
# Activate the uv-managed venv (created once — see README "Python environments"):
# source "${REPO_DIR}/.venvs/si_curriculum/bin/activate"

cd "${REPO_DIR}/3_si_curriculum"

python curriculum_generator/verify_questions.py \
    --input_json  "${CURRICULUM_OUTPUT}/curriculum.json" \
    --output_json "${VERIFIED_OUTPUT}/curriculum_verified.json" \
    --model_ids   ${MODEL_IDS}
