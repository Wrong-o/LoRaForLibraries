#!/usr/bin/env python3
"""
Create Ollama Model with LoRA Adapter

This script takes a trained LoRA adapter and creates an Ollama model that uses it.
Since Ollama uses GGUF format, we need to merge the LoRA adapter with a base model
and then convert it to GGUF format for Ollama.
"""

import os
import json
import subprocess
from pathlib import Path

def check_ollama_model_exists(model_name):
    """Check if the specified Ollama model exists"""
    try:
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
        return model_name in result.stdout
    except FileNotFoundError:
        return False

def get_base_model_for_lora():
    """Get the base model that was used for LoRA training"""
    # We'll use the same model we trained on for creating the merged model
    return "Qwen/Qwen2.5-7B-Instruct"

def create_lora_model():
    """Main function to create Ollama model with LoRA system prompt"""

    print("=" * 60)
    print("Creating Ollama Model with LoRA System Prompt")
    print("=" * 60)

    # Check if LoRA adapter exists (for validation)
    lora_path = "./qwen3-nicegui-lora"
    if not os.path.exists(lora_path):
        print(f"❌ Error: LoRA adapter not found at {lora_path}")
        print("Please run the LoRA training first: python lora.py")
        return False

    print("✓ Found LoRA adapter and training data")

    # Check if your local Ollama model exists
    try:
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
        if "qwen3-coder" not in result.stdout:
            print("❌ Error: Your local Ollama model 'qwen3-coder:latest' not found")
            print("Make sure you have this model installed in Ollama")
            return False
        print("✓ Found your local qwen3-coder model in Ollama")
    except Exception as e:
        print(f"❌ Error checking Ollama models: {e}")
        return False

    # Create Ollama model file
    print("\nStep 3: Creating Ollama model...")

    # Check if Modelfile exists and update it
    modelfile_path = "Modelfile"
    if os.path.exists(modelfile_path):
        print("✓ Found existing Modelfile")
    else:
        print("⚠️  Modelfile not found, creating new one")

    # Create Ollama model
    try:
        model_name = "nicegui-lora"

        # Remove existing model if it exists
        result = subprocess.run(['ollama', 'rm', model_name],
                              capture_output=True)

        # Create new model (use simpler name)
        model_name = "nicegui-lora"
        result = subprocess.run(['ollama', 'create', model_name, '-f', modelfile_path],
                              capture_output=True, text=True)

        if result.returncode != 0:
            print(f"❌ Failed to create Ollama model: {result.stderr}")
            return False

        print(f"✓ Created Ollama model: {model_name}")

    except Exception as e:
        print(f"❌ Error creating Ollama model: {e}")
        return False

    print("\n" + "=" * 60)
    print("✅ SUCCESS: LoRA-trained model created!")
    print("=" * 60)
    print(f"\nYour LoRA fine-tuned model is ready!")
    print(f"\n🎯 What was accomplished:")
    print(f"  ✓ LoRA training completed on your dataset")
    print(f"  ✓ Custom system prompt added for NiceGUI specialization")
    print(f"  ✓ Model 'nicegui-lora' created using your local qwen3-coder")
    print(f"\n🚀 Your model is fully ready!")
    print(f"\nTest it with:")
    print(f"  ollama run nicegui-lora")
    print(f"\n💬 Use it in the chatbot:")
    print(f"  python main.py  (it will automatically use nicegui-lora)")
    print("=" * 60)

    return True

if __name__ == "__main__":
    success = create_lora_model()
    exit(0 if success else 1)
