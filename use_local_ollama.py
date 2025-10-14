#!/usr/bin/env python3
"""
Use Local Ollama Model for NiceGUI Chatbot

This script bypasses the training process and directly uses your locally installed
Ollama model with the NiceGUI chatbot interface.
"""

import os
import subprocess

def get_ollama_models():
    """Get list of available Ollama models"""
    try:
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
        if result.returncode != 0:
            return None
        return result.stdout.strip()
    except FileNotFoundError:
        return None

print("=" * 50)
print("NiceGUI with Local Ollama Model")
print("=" * 50)

# Check if Ollama is installed
models_output = get_ollama_models()
if models_output is None:
    print("❌ Error: Ollama not found or not running")
    print("Please install Ollama from https://ollama.ai/")
    exit(1)

print("✓ Ollama is installed")
print("\nAvailable models:")
print(models_output)

# Ask user which model to use
model_name = input("\nEnter the name of the Ollama model to use (e.g., qwen3-coder:latest): ")
if not model_name:
    model_name = "qwen3-coder:latest"
    print(f"Using default model: {model_name}")

# Validate and suggest similar model names
if model_name not in models_output:
    print(f"\n⚠️  Warning: '{model_name}' not found in available models.")
    print("Available models are:")

    available_models = []
    for line in models_output.split('\n'):
        if line.strip() and not line.startswith('NAME'):
            parts = line.split()
            if parts:
                model_name_available = parts[0]
                available_models.append(model_name_available)
                print(f"  - {model_name_available}")

    # Try to find similar model names
    import difflib
    close_matches = difflib.get_close_matches(model_name, available_models, n=3, cutoff=0.6)

    if close_matches:
        print(f"\nDid you mean one of these?")
        for match in close_matches:
            print(f"  - {match}")

        # Ask if they want to use the closest match
        use_suggestion = input(f"\nUse '{close_matches[0]}' instead? (y/n): ")
        if use_suggestion.lower() == 'y':
            model_name = close_matches[0]
            print(f"Using '{model_name}' instead.")
        else:
            print(f"Continuing with '{model_name}' as specified...")
    else:
        print(f"\nNo similar models found. Continuing with '{model_name}'...")

# Create symlink to main.py
print(f"\nSetting up NiceGUI chatbot with model: {model_name}")

# Modify the main.py to use the specified model
with open('main.py', 'r') as f:
    content = f.read()

# Replace the model name in main.py
import re
updated_content = re.sub(r'OLLAMA_MODEL = ".*"', f'OLLAMA_MODEL = "{model_name}"', content)

# Write the updated content to a temporary file
with open('main_temp.py', 'w') as f:
    f.write(updated_content)

# Replace the original file
os.replace('main_temp.py', 'main.py')

print(f"\n✓ Setup complete!")
print(f"The chatbot is configured to use: {model_name}")
print(f"\nTo start the chatbot, run:")
print(f"  python main.py")
print(f"\nThe chatbot will be available at: http://localhost:7860")

start_now = input("\nStart the chatbot now? (y/n): ")
if start_now.lower() == 'y':
    print("\n" + "=" * 50)
    print("Starting NiceGUI chatbot...")
    print(f"Using model: {model_name}")
    print("The chatbot will run on http://localhost:7860")
    print("Press Ctrl+C to stop the server")
    print("=" * 50)
    os.system("python main.py")
else:
    print(f"\nYou can start the chatbot later with: python main.py")
