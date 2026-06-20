#!/bin/bash
# Setup script for rl_trainer conda environment
# Run from della: bash setup_rl_env.sh

set -e

ENV_NAME="rl_trainer"

echo "============================================"
echo "Setting up $ENV_NAME conda environment"
echo "============================================"

module purge
module load anaconda3/2024.6

# Remove existing env if it exists
if conda env list | grep -q "$ENV_NAME"; then
    echo "Removing existing $ENV_NAME environment..."
    conda env remove -n $ENV_NAME -y
fi

# Create fresh env with Python 3.10
echo "Creating conda environment: $ENV_NAME"
conda create -n $ENV_NAME python=3.10 -y

# Activate
source activate $ENV_NAME

# PyTorch with CUDA 12.1 support (matches della's CUDA)
echo "Installing PyTorch with CUDA 12.1..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Core training stack
echo "Installing core training libraries..."
pip install \
    transformers>=4.46.0 \
    accelerate>=0.34.0 \
    deepspeed>=0.15.0 \
    trl>=0.12.0 \
    peft>=0.13.0

# Data processing
echo "Installing data processing libraries..."
pip install \
    datasets>=2.20.0 \
    numpy>=1.24.0

# Tokenizers and model support
echo "Installing model support libraries..."
pip install \
    sentencepiece \
    protobuf \
    safetensors

# Experiment tracking (optional but useful)
echo "Installing utilities..."
pip install wandb

# Verify installations
echo ""
echo "============================================"
echo "Verifying installations..."
echo "============================================"
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU count: {torch.cuda.device_count()}')

import transformers
print(f'Transformers: {transformers.__version__}')

import accelerate
print(f'Accelerate: {accelerate.__version__}')

import deepspeed
print(f'DeepSpeed: {deepspeed.__version__}')

import trl
print(f'TRL: {trl.__version__}')

import peft
print(f'PEFT: {peft.__version__}')

import datasets
print(f'Datasets: {datasets.__version__}')

# Verify GRPOTrainer imports cleanly
from trl import GRPOConfig, GRPOTrainer
print('GRPOTrainer imported successfully!')
"

echo ""
echo "============================================"
echo "Environment $ENV_NAME is ready!"
echo "Update your SLURM script: conda activate $ENV_NAME"
echo "============================================"