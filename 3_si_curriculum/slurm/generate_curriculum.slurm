#!/bin/bash
#SBATCH --job-name=si-curriculum
#SBATCH --output=logs/curriculum_%j.out
#SBATCH --error=logs/curriculum_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
# #SBATCH --mail-type=END,FAIL
# #SBATCH --mail-user=your@email.com

# ============================================================
# SI Curriculum Generation
# Generates multi-hop Q&A items from GraphMERT-expanded KG.
#
# Required env vars (set before sbatch, or edit below):
#   REPO_DIR       - root of neuro_SI_pipeline checkout
#   OUTPUT_BASE    - scratch output directory
#   MANIFEST_PATH  - path to kg_manifest.json from calculate_hops.py
#   OPENAI_API_KEY - API key for QA generation LLM
# ============================================================

set -euo pipefail

: "${REPO_DIR:?REPO_DIR must be set}"
: "${OUTPUT_BASE:?OUTPUT_BASE must be set}"
: "${MANIFEST_PATH:?MANIFEST_PATH must be set}"

CURRICULUM_OUTPUT="${OUTPUT_BASE}/curriculum"
mkdir -p "${CURRICULUM_OUTPUT}" logs

module purge
# Activate the uv-managed venv (created once — see README "Python environments"):
# source "${REPO_DIR}/.venvs/si_curriculum/bin/activate"

cd "${REPO_DIR}/3_si_curriculum"

python curriculum_generator/generate_curriculum.py \
    --manifest_path "${MANIFEST_PATH}" \
    --output_dir    "${CURRICULUM_OUTPUT}" \
    --min_hops      3 \
    --max_hops      5 \
    --target_count  50000 \
    ${OPENAI_API_KEY:+--api_key "${OPENAI_API_KEY}"}
