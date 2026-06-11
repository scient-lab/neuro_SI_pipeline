#!/bin/bash
#SBATCH --job-name=entity_discovery
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16G
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --time=04:00:00
#SBATCH --mail-type=fail
# SBATCH --mail-user=YOUR_EMAIL@YOUR_INSTITUTION.EDU
#SBATCH --output=slurm_logs/entity_discovery_%j.out

module purge
# Activate the uv-managed venv (created once — see README "Python environments")
source ${REPO_DIR}/.venvs/graphmert/bin/activate

# Set these before submitting:
#   export REPO_DIR=/path/to/neuro_SI_pipeline
#   export OUTPUT_BASE=/scratch/gpfs/$USER/neuro_pipeline
#   export MODEL_ID=/path/to/qwen3-32b

cd ${REPO_DIR}/2_graphmert

python utils/entity_discovery/entity_discovery.py \
  --tokenized_dir  ${OUTPUT_BASE}/graphmert/tokenized_inputs/train_tokenized \
  --output_dir     ${OUTPUT_BASE}/graphmert/dataset_with_heads/chunks \
  --model_id       ${MODEL_ID} \
  --tokenizer      ${OUTPUT_BASE}/graphmert/stable_tokenizer
