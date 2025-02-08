#!/bin/bash

# Check if ollama is installed
if ! command -v ollama &> /dev/null; then
    echo "Error: ollama is not installed"
    echo "Please install ollama first: curl -fsSL https://ollama.com/install.sh | sh"
    exit 1
fi

# List of models to download
declare -a models=(
    "qwen2.5-coder:3b"
    "deepseek-r1:1.5b"
    "qwen2.5-coder:14b"
    "qwen2.5-coder:1.5b"
    "qwen2.5-coder:0.5b"
    "deepseek-r1:32b"
)

echo "Starting model downloads..."

# Loop through models and pull each one
for model in "${models[@]}"; do
    echo "Downloading $model..."
    if ollama pull "$model"; then
        echo "Successfully downloaded $model"
    else
        echo "Failed to download $model"
    fi
done

echo "Verifying installation..."
ollama list

echo "Installation complete!"