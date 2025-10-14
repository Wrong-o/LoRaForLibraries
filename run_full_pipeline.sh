#!/bin/bash
# Complete LoRA training pipeline for NiceGUI
# This script runs all steps sequentially

set -e  # Exit on any error

echo "=================================================="
echo "NiceGUI LoRA Fine-tuning - Complete Pipeline"
echo "=================================================="
echo ""

echo "This will take 3-7 hours depending on your GPU."
echo "Steps:"
echo "  1. Train LoRA adapter on compatible model (Qwen2.5-7B)"
echo "  2. Merge LoRA with base model"
echo "  3. Convert merged model to GGUF format"
echo "  4. Create Ollama model with your local qwen3-coder as base"
echo ""
echo "Note: This process uses a compatible model for training"
echo "      but creates a model that works with your local Ollama qwen3-coder"
echo ""
read -p "Continue with full training pipeline? (y/N) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 0
fi

echo ""
echo "Starting pipeline..."
echo ""

# Step 1: Train
echo "STEP 1/4: Training LoRA adapter..."
python train.py
if [ $? -ne 0 ]; then
    echo "❌ Training failed"
    exit 1
fi

# Step 2: Merge
echo ""
echo "STEP 2/4: Merging LoRA with base model..."
python merge_lora.py
if [ $? -ne 0 ]; then
    echo "❌ Merging failed"
    exit 1
fi

# Step 3: Convert
echo ""
echo "STEP 3/4: Converting to GGUF..."
bash convert_to_gguf.sh
if [ $? -ne 0 ]; then
    echo "❌ Conversion failed"
    exit 1
fi

# Step 4: Create Ollama model with LoRA
echo ""
echo "STEP 4/4: Creating Ollama model with LoRA adapter..."
python create_ollama_lora_model.py
if [ $? -ne 0 ]; then
    echo "❌ Ollama model creation failed"
    exit 1
fi

echo ""
echo "=================================================="
echo "✓ Complete Pipeline Finished!"
echo "=================================================="
echo ""
echo "Your LoRA fine-tuned model is ready!"
echo ""
echo "Test it:"
echo "  ollama run qwen3-nicegui-lora"
echo ""
echo "Or use the chatbot:"
echo "  python main.py"
echo ""
echo "=================================================="


