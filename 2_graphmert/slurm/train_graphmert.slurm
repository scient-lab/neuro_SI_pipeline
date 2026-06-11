#!/bin/bash
#SBATCH --job-name=graphmert_mlm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16G
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --time=12:00:00
#SBATCH --mail-type=begin,end,fail
# SBATCH --mail-user=YOUR_EMAIL@YOUR_INSTITUTION.EDU
#SBATCH --output=slurm_logs/train_graphmert_%j.out

module purge
# Activate the uv-managed venv (created once — see README "Python environments")
source ${REPO_DIR}/.venvs/graphmert/bin/activate

# Set these before submitting:
#   export REPO_DIR=/path/to/neuro_SI_pipeline
#   export OUTPUT_BASE=/scratch/gpfs/$USER/neuro_pipeline

cd ${REPO_DIR}/2_graphmert

# Edit launch_configs/args_mlm.yaml first to set all paths correctly,
# then submit this script.
python run_mlm.py --yaml_file launch_configs/args_mlm.yaml

# Usage:
#   sbatch slurm/train_graphmert.slurm
#   squeue -u $USER
#   scancel <job_id>
