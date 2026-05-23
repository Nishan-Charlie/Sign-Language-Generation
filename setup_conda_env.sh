#!/bin/bash

# Exit on any error
set -e

# 1. Path to your Miniconda installation (detected in /home/e21283/miniconda3)
CONDA_PATH="/home/e21283/miniconda3"
ENV_NAME="sinhala_ssl"

echo "--- Initializing Conda from $CONDA_PATH ---"

# Source the conda setup script to make the 'conda' command available in this shell
# This is necessary because the shell might not have it in its standard PATH yet.
if [ -f "$CONDA_PATH/etc/profile.d/conda.sh" ]; then
    source "$CONDA_PATH/etc/profile.d/conda.sh"
else
    echo "Error: Could not find conda.sh in $CONDA_PATH/etc/profile.d/"
    echo "Please check your miniconda3 installation path."
    exit 1
fi

# 2. Create the Conda Environment
# Using Python 3.10 for stable compatibility with mT5 and Transformers
echo "Creating conda environment: $ENV_NAME with Python 3.10..."
conda create -y -n $ENV_NAME python=3.10

# 3. Activate the Environment
echo "Activating environment..."
conda activate $ENV_NAME

# 4. Install PyTorch with CUDA support (adjusting for common GPU setups)
# We'll install pytorch and related tools from the pytorch channel for better GPU optimization
echo "Installing PyTorch (with CUDA support)..."
conda install -y pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# 5. Install Hugging Face and ML Libraries via pip (to ensure latest versions)
echo "Installing Transformers, Datasets, and other ML tools..."
pip install transformers datasets accelerate evaluate \
            sentencepiece protobuf rouge_score sacrebleu pandas

echo "-------------------------------------------------------"
echo "Success! Conda environment '$ENV_NAME' is ready."
echo "To use it in your terminal, run:"
echo "    conda activate $ENV_NAME"
echo "-------------------------------------------------------"
