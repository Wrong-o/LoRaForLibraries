#!/usr/bin/env python3
"""
Training Pipeline Runner
This script orchestrates the complete training pipeline:
1. Verify dataset exists
2. Train LoRA adapter
3. Provide next steps
"""

import os
import sys
import subprocess

def check_file_exists(filepath, description):
    """Check if a required file exists."""
    if not os.path.exists(filepath):
        print(f"❌ Error: {description} not found at: {filepath}")
        return False
    return True

def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{'=' * 50}")
    print(f"{description}")
    print(f"{'=' * 50}\n")
    
    result = subprocess.run(cmd, shell=True)
    
    if result.returncode != 0:
        print(f"\n❌ Error: {description} failed with exit code {result.returncode}")
        return False
    
    return True

def main():
    print("=" * 50)
    print("NiceGUI LoRA Training Pipeline")
    print("=" * 50)
    
    # Step 1: Verify dataset exists
    print("\n📋 Step 1: Verifying dataset...")
    if not check_file_exists("nicegui_lora_dataset.json", "Training dataset"):
        print("\nTo create the dataset, run:")
        print("  python download_examples.py")
        sys.exit(1)
    
    # Count examples in dataset
    import json
    with open("nicegui_lora_dataset.json", "r") as f:
        dataset = json.load(f)
    print(f"✓ Dataset found with {len(dataset)} examples")
    
    # Step 2: Check if already trained
    if os.path.exists("qwen3-nicegui-lora"):
        print("\n⚠️  Warning: LoRA adapter already exists at ./qwen3-nicegui-lora")
        response = input("Do you want to retrain? This will overwrite existing weights. (y/N): ")
        if response.lower() != 'y':
            print("Skipping training. Using existing adapter.")
            print("\nTo merge the model, run:")
            print("  python merge_lora.py")
            sys.exit(0)
    
    # Step 3: Train LoRA
    print("\n🚀 Step 2: Starting LoRA training...")
    print("This will take several hours depending on your GPU.")
    print("Training parameters:")
    print("  - Model: Qwen2.5-Coder-7B-Instruct")
    print("  - Method: QLoRA (4-bit)")
    print("  - Epochs: 3")
    print("  - Batch size: 1 x 8 (gradient accumulation)")
    print("")
    
    if not run_command("python lora.py", "LoRA Training"):
        sys.exit(1)
    
    # Step 4: Provide next steps
    print("\n" + "=" * 50)
    print("✓ Training Pipeline Complete!")
    print("=" * 50)
    print("\nNext steps:")
    print("  1. Merge LoRA adapter with base model:")
    print("     python merge_lora.py")
    print("")
    print("  2. Convert to GGUF format for Ollama:")
    print("     bash convert_to_gguf.sh")
    print("")
    print("  3. Create Ollama model:")
    print("     ollama create qwen3-nicegui -f Modelfile")
    print("")
    print("  4. Update main.py to use your fine-tuned model")
    print("=" * 50)

if __name__ == "__main__":
    main()


