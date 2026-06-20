#!/bin/bash
#SBATCH --job-name=rl_v3_smoke_test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --constraint=gpu80
#SBATCH --mem=512G
#SBATCH --time=01:00:00
#SBATCH --mail-type=all
# #SBATCH --mail-user=YOUR_EMAIL@YOUR_INSTITUTION.EDU
#SBATCH -o slurm_logs/rl_v3_smoke_%j.out
#SBATCH -e slurm_logs/rl_v3_smoke_%j.err

# Required env vars:
#   REPO_DIR      - root of neuro_SI_pipeline checkout
#   MODEL_NAME    - path to SFT-merged checkpoint
#   DATASET_PATH  - path to prepared RL dataset
#   OUTPUT_BASE   - scratch output directory (smoke test writes to ${OUTPUT_BASE}/_v3_smoke_test)

: "${REPO_DIR:?REPO_DIR must be set}"
: "${MODEL_NAME:?MODEL_NAME must be set}"
: "${DATASET_PATH:?DATASET_PATH must be set}"
: "${OUTPUT_BASE:?OUTPUT_BASE must be set}"

module purge
module load anaconda3/2024.6
conda activate rl_trainer

export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500
export NCCL_DEBUG=WARN
export CUDA_VISIBLE_DEVICES=0,1,2,3
export CUDA_MODULE_LOADING=EAGER
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,garbage_collection_threshold:0.7"
export OMP_NUM_THREADS=12
export MKL_NUM_THREADS=12
export CUDA_HOME=/usr/local/cuda-12.8
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export HF_HOME=~/.cache/huggingface

cd "${REPO_DIR}/3_si_curriculum/RL"
mkdir -p slurm_logs

echo "========================================="
echo " V3 SMOKE TEST (20 steps, ~55 min)"
echo " Tests full V3 config end-to-end"
echo ""
echo " Validates:"
echo "   - Gated path_alignment reward"
echo "   - Smooth length penalty (tokenizer-based)"
echo "   - log_completions=True"
echo "   - GenerationDumpCallback (JSONL)"
echo "   - HeldOutEvalCallback (runs 2x)"
echo "   - save_strategy=steps (fires 2x)"
echo "   - beta=0.08, max_completion=1280"
echo ""
echo " Node: $(hostname)"
echo "========================================="

torchrun --nnodes=1 --nproc_per_node=4 \
    --rdzv_id=100 --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    test_rl.py

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "========================================="
    echo " SMOKE TEST EXITED CLEANLY"
    echo ""
    echo " CHECK THESE IN THE OUTPUT DIRECTORY:"
    echo "   1. training_generations.jsonl exists + has entries"
    echo "   2. eval_history.json exists with 2 eval entries"
    echo "   3. checkpoint-10/ exists with model weights"
    echo "   4. Reward values look sane in logs"
    echo "   5. Completion lengths stay under ~700 tokens"
    echo ""
    echo " Look for 'STATUS: PASS' line in slurm stdout"
    echo " If anything missing, DO NOT launch full run"
    echo "========================================="
    rm -rf "${OUTPUT_BASE}/_v3_smoke_test"
    echo "Cleaned up test checkpoint directory."
else
    echo "========================================="
    echo " SMOKE TEST FAILED (exit code $EXIT_CODE)"
    echo " Check stderr log for traceback"
    echo " DO NOT launch the full run until fixed"
    echo " Keeping test checkpoint at: ${OUTPUT_BASE}/_v3_smoke_test"
    echo "========================================="
fi
