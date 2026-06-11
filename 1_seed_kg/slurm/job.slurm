#!/bin/bash
#SBATCH --job-name=seed_kg_extraction
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=20G
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --time=04:00:00
# SBATCH --array=0-3      # uncomment and set based on count_text_units.py output
#SBATCH --mail-type=begin,end,fail
# SBATCH --mail-user=YOUR_EMAIL@YOUR_INSTITUTION.EDU
#SBATCH --output=slurm_logs/%A_%a.out

module purge
# Activate the uv-managed venv (created once — see README "Python environments")
source ${REPO_DIR}/.venvs/graphrag/bin/activate

# Set these before submitting:
#   export REPO_DIR=/path/to/neuro_SI_pipeline
#   export OUTPUT_BASE=/scratch/gpfs/$USER/neuro_pipeline
#   export MODEL_ID=/path/to/your/llm/model

cd ${REPO_DIR}/1_seed_kg

python graphrag_index.py \
  --root_dir  ${OUTPUT_BASE}/graphrag \
  --model_id  ${MODEL_ID} \
  --step      3

# Usage:
#   sbatch slurm/job.slurm
#   squeue -u $USER
#   scancel <job_id>
