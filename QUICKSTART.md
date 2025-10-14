# Quick Start Guide

## TL;DR

Train a NiceGUI-specialized coding assistant in 4 commands:

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train the model (2-4 hours)
python train.py

# 3. Merge and convert
python merge_lora.py
bash convert_to_gguf.sh

# 4. Create Ollama model
ollama create qwen3-nicegui -f Modelfile

# 5. Use it!
python main.py
# Or: ollama run qwen3-nicegui
```

## What You Get

A specialized LLM that:
- Knows NiceGUI syntax inside and out
- Provides accurate code examples
- Understands NiceGUI best practices
- Runs locally on your machine

## Training Time

- **RTX 4090**: ~4-6 hours total
- **RTX 3090**: ~8-12 hours total

## Verification

Test the model:

```bash
ollama run qwen3-nicegui "How do I create a button in NiceGUI?"
```

Expected response should include proper NiceGUI code with `ui.button()`.

## Files Overview

| File | Purpose |
|------|---------|
| `train.py` | Main training pipeline (run this first) |
| `lora.py` | LoRA training script (called by train.py) |
| `merge_lora.py` | Merge LoRA adapter with base model |
| `convert_to_gguf.sh` | Convert to Ollama-compatible format |
| `main.py` | NiceGUI chatbot interface |
| `Modelfile` | Ollama model configuration |

## Troubleshooting

**Out of memory?**
```python
# Edit lora.py, change line 85:
gradient_accumulation_steps=16  # increase from 8
```

**Model not found?**
```bash
ollama list  # check if model is registered
ollama create qwen3-nicegui -f Modelfile  # re-create
```

**Want to retrain?**
```bash
rm -rf qwen3-nicegui-lora/  # delete old adapter
python train.py  # train again
```

For detailed instructions, see [SETUP.md](SETUP.md).


