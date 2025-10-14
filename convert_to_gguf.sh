#!/bin/bash

echo "=================================================="
echo "GGUF Conversion Script for Ollama"
echo "=================================================="

MERGED_MODEL_PATH="./qwen3-nicegui-merged"
OUTPUT_NAME="qwen3-nicegui"

# Check if merged model exists
if [ ! -d "$MERGED_MODEL_PATH" ]; then
    echo "❌ Error: Merged model not found at $MERGED_MODEL_PATH"
    echo "Please run: python merge_lora.py first"
    exit 1
fi

# Check if llama.cpp is installed and built
if [ ! -d "llama.cpp" ]; then
    echo "📦 Cloning llama.cpp repository..."
    git clone https://github.com/ggerganov/llama.cpp.git
fi

cd llama.cpp

# Check if binaries exist
if [ ! -f "llama-quantize" ] || [ ! -f "convert_hf_to_gguf.py" ]; then
    echo "🔨 Building llama.cpp with required tools..."
    # Build with make - first try to build all tools
    make clean
    make -j$(nproc) llama-quantize convert_hf_to_gguf.py

    # If that doesn't work, try cmake build
    if [ ! -f "llama-quantize" ]; then
        echo "🔨 Trying cmake build..."
        mkdir -p build
        cd build
        cmake ..
        make -j$(nproc) llama-quantize
        cd ..
    fi
else
    echo "✓ llama.cpp tools already built"
fi

cd ..

# Create output directory
mkdir -p ./gguf_models

echo ""
echo "Step 1: Converting model to GGUF format..."
python3 llama.cpp/convert_hf_to_gguf.py \
    "$MERGED_MODEL_PATH" \
    --outfile "./gguf_models/${OUTPUT_NAME}.gguf" \
    --outtype f16

if [ $? -ne 0 ]; then
    echo "❌ Conversion failed"
    exit 1
fi

echo ""
echo "Step 2: Quantizing to Q4_K_M (recommended for Ollama)..."
llama.cpp/llama-quantize \
    "./gguf_models/${OUTPUT_NAME}.gguf" \
    "./gguf_models/${OUTPUT_NAME}-q4_k_m.gguf" \
    Q4_K_M

if [ $? -ne 0 ]; then
    echo "❌ Quantization failed"
    exit 1
fi

echo ""
echo "=================================================="
echo "✓ Conversion complete!"
echo "=================================================="
echo ""
echo "GGUF model saved to: ./gguf_models/${OUTPUT_NAME}-q4_k_m.gguf"
echo ""
echo "To load this model in Ollama, create a Modelfile:"
echo ""
echo "--- Modelfile ---"
echo "FROM ./gguf_models/${OUTPUT_NAME}-q4_k_m.gguf"
echo ""
echo "TEMPLATE \"\"\"{{ if .System }}<|im_start|>system"
echo "{{ .System }}<|im_end|>"
echo "{{ end }}{{ if .Prompt }}<|im_start|>user"
echo "{{ .Prompt }}<|im_end|>"
echo "{{ end }}<|im_start|>assistant"
echo "{{ .Response }}<|im_end|>"
echo "\"\"\""
echo ""
echo "PARAMETER stop <|im_start|>"
echo "PARAMETER stop <|im_end|>"
echo "--- End of Modelfile ---"
echo ""
echo "Then run:"
echo "  ollama create ${OUTPUT_NAME} -f Modelfile"
echo "  ollama run ${OUTPUT_NAME}"
echo ""
echo "=================================================="


