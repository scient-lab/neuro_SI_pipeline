#!/bin/bash
#SBATCH --job-name=predict_tails
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16G
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --time=06:00:00
#SBATCH --array=0-3       # 4 shards — adjust based on dataset size
#SBATCH --mail-type=fail
# SBATCH --mail-user=YOUR_EMAIL@YOUR_INSTITUTION.EDU
#SBATCH --output=slurm_logs/predict_tails_%A_%a.out

module purge
# Activate the uv-managed venv (created once — see README "Python environments")
source ${REPO_DIR}/.venvs/graphmert/bin/activate

# Set these before submitting:
#   export REPO_DIR=/path/to/neuro_SI_pipeline
#   export OUTPUT_BASE=/scratch/gpfs/$USER/neuro_pipeline
#   export MODEL_ID=/path/to/your/qwen3-32b

cd ${REPO_DIR}/2_graphmert

python predict_tails_llm.py \
  --model_id   ${MODEL_ID} \
  --tokenizer  ${OUTPUT_BASE}/graphmert/stable_tokenizer \
  --dataset    ${OUTPUT_BASE}/graphmert/llm_relations/relations_cleaned_train \
  --output_dir ${OUTPUT_BASE}/graphmert/graphmert_kg/predictions \
  --num_shards 4
  # --shard_id is auto-detected from SLURM_ARRAY_TASK_ID
