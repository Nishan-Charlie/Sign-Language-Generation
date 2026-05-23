#!/bin/bash

# Exit on any error
set -e

echo "--- Starting Environment Setup for Sinhala mT5 Training ---"

# 1. Define Environment Name
ENV_NAME="venv_sinhala"

# 2. Check if python3 is installed
if ! command -v python3 &> /dev/null
then
    echo "Error: python3 could not be found. Please install it first."
    exit 1
fi

# 3. Create Virtual Environment
echo "Creating virtual environment: $ENV_NAME..."
python3 -m venv $ENV_NAME

# 4. Activate Environment
echo "Activating environment..."
source $ENV_NAME/bin/activate

# 5. Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# 6. Install Core Libraries
echo "Installing base machine learning libraries (torch, transformers)..."
# Using standard pip install. For specific CUDA versions, use e.g. 'pip install torch --index-url https://download.pytorch.org/whl/cu118'
pip install torch transformers datasets accelerate evaluate

# 7. Install Tokenizer and Metric Dependencies
echo "Installing tokenizer and metric dependencies..."
pip install sentencepiece protobuf rouge_score sacrebleu pandas

echo "-------------------------------------------------------"
echo "Success! Environment '$ENV_NAME' is ready."
echo "To activate it in your terminal, run:"
echo "    source $ENV_NAME/bin/activate"
echo "-------------------------------------------------------"
