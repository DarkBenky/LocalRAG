
# List of models to download
models=(
  "qwen2.5-coder:3b"
  "deepseek-r1:1.5b"
  "qwen2.5-coder:14b"
  "qwen2.5-coder:1.5b"
  "qwen2.5-coder:0.5b"
  "deepseek-r1:32b"
)

# Loop through models and pull each one
for model in "${models[@]}"
do
  ollama pull $model
done

# Verify installation
ollama list
