#!/usr/bin/env python3
"""
Merge LoRA adapter with base model
This script loads the trained LoRA adapter and merges it with the base model,
creating a single unified model that can be converted to GGUF format.
"""

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch
import os

print("=" * 50)
print("LoRA Model Merger")
print("=" * 50)

# Configuration
BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"  # Compatible model for LoRA merging
LORA_ADAPTER_PATH = "./qwen3-nicegui-lora"
OUTPUT_PATH = "./qwen3-nicegui-merged"

# Verify LoRA adapter exists
if not os.path.exists(LORA_ADAPTER_PATH):
    print(f"\n❌ Error: LoRA adapter not found at {LORA_ADAPTER_PATH}")
    print("Please train the model first using: python lora.py")
    exit(1)

print(f"\n1. Loading base model for merging: {BASE_MODEL}")
print("Note: Using compatible model for LoRA merging")

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
)

print(f"\n2. Loading LoRA adapter from: {LORA_ADAPTER_PATH}")
model = PeftModel.from_pretrained(base_model, LORA_ADAPTER_PATH)

print("\n3. Merging LoRA weights into base model...")
model = model.merge_and_unload()

print(f"\n4. Saving merged model to: {OUTPUT_PATH}")
os.makedirs(OUTPUT_PATH, exist_ok=True)
model.save_pretrained(OUTPUT_PATH, safe_serialization=True)
tokenizer.save_pretrained(OUTPUT_PATH)

print("\n✓ Successfully merged model!")
print(f"   Saved to: {OUTPUT_PATH}")
print("\n5. Next step: Convert to GGUF format")
print("   Run: bash convert_to_gguf.sh")
print("=" * 50)


