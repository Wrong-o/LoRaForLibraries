# NiceGUI LoRA Fine-tuning Setup Guide

This project fine-tunes a large language model (LLM) specifically on NiceGUI examples to create an expert coding assistant for NiceGUI development.

## Overview

The pipeline consists of:
1. Download NiceGUI examples from GitHub
2. Train a LoRA adapter on these examples
3. Merge the adapter with the base model
4. Convert to GGUF format for Ollama
5. Deploy in the NiceGUI chatbot

## Hardware Requirements

- **GPU**: NVIDIA GPU with at least 24GB VRAM (RTX 3090)
- **RAM**: 32GB+ system RAM recommended
- **Storage**: ~50GB free space for models and temporary files
- **OS**: Linux (recommended), Windows with WSL2, or macOS with Metal support

## Software Requirements

- Python 3.10+
- CUDA 12.0+ (for NVIDIA GPUs)
- Git
- Ollama (for inference)

## Installation

### 1. Clone and Setup Environment

```bash
cd /home/jayb/projects/nicegui
pip install -r requirements.txt
```

### 2. Download NiceGUI Examples (Optional)

If you want to refresh the dataset:

```bash
python download_examples.py
```

This will create `nicegui_lora_dataset.json` with ~300+ NiceGUI code examples.

## Training Pipeline

### Quick Start (Automated)

Run the complete training pipeline with one command:

```bash
python train.py
```

This script will:
- Verify the dataset exists
- Train the LoRA adapter (~2-4 hours on RTX 4090)
- Provide instructions for next steps

### Manual Steps (Advanced)

If you prefer to run each step manually:

#### Step 1: Train LoRA Adapter

```bash
python lora.py
```

**What this does:**
- Loads Qwen2.5-Coder-7B-Instruct model
- Applies QLoRA (4-bit quantization)
- Trains LoRA adapters (rank 16) for 3 epochs
- Saves adapter weights to `./qwen3-nicegui-lora`

**Expected time:** 2-4 hours on RTX 4090, 4-8 hours on RTX 3090

**Training progress:**
- You'll see loss decreasing over time
- Evaluation metrics every 50 steps
- Model checkpoints saved every 100 steps

#### Step 2: Merge LoRA with Base Model

```bash
python merge_lora.py
```

**What this does:**
- Loads the base model and LoRA adapter
- Merges the adapter weights into the base model
- Saves the unified model to `./qwen3-nicegui-merged`

**Expected time:** 5-10 minutes

#### Step 3: Convert to GGUF Format

```bash
bash convert_to_gguf.sh
```

**What this does:**
- Clones llama.cpp (if not present)
- Converts merged model to GGUF format
- Quantizes to Q4_K_M (optimal for performance/quality)
- Saves to `./gguf_models/qwen3-nicegui-q4_k_m.gguf`

**Expected time:** 10-20 minutes

#### Step 4: Create Ollama Model

```bash
ollama create qwen3-nicegui -f Modelfile
```

**What this does:**
- Registers the GGUF model with Ollama
- Applies the template and system prompt
- Makes the model available for inference

#### Step 5: Test the Model

```bash
ollama run qwen3-nicegui
```

Try asking: "How do I create a button in NiceGUI?"

## Using the Fine-tuned Model in the Chatbot

The `main.py` chatbot is already configured to use the fine-tuned model. Once you've completed the training pipeline:

```bash
python main.py
```

The chatbot will automatically use `qwen3-nicegui` model and should provide expert-level NiceGUI assistance.

## Troubleshooting

### Out of Memory (OOM) Errors

If you get OOM errors during training:

1. **Reduce batch size** in `lora.py`:
   ```python
   gradient_accumulation_steps=16  # increase from 8
   ```

2. **Reduce max sequence length**:
   ```python
   max_length=768  # reduce from 1024
   ```

3. **Use smaller LoRA rank**:
   ```python
   r=8  # reduce from 16
   ```

### Training is too slow

- Ensure you're using GPU (check with `nvidia-smi`)
- Close other GPU-intensive applications
- Consider using a smaller model (already using 7B)

### Conversion fails

- Ensure llama.cpp is built successfully
- Check that merged model has all required files
- Try manual conversion with custom parameters

### Model not found in Ollama

```bash
ollama list  # verify model is registered
ollama create qwen3-nicegui -f Modelfile  # re-create if needed
```

### Poor model quality

If the fine-tuned model doesn't perform well:

1. **Increase training epochs** (in `lora.py`):
   ```python
   num_train_epochs=5  # increase from 3
   ```

2. **Adjust learning rate**:
   ```python
   learning_rate=2e-4  # reduce if unstable
   ```

3. **Add more training data**: Download additional examples or create custom ones

## Project Structure

```
nicegui/
├── download_examples.py      # Download NiceGUI examples
├── nicegui_lora_dataset.json # Training dataset (~300 examples)
├── lora.py                   # LoRA training script
├── merge_lora.py             # Merge LoRA with base model
├── convert_to_gguf.sh        # Convert to GGUF format
├── train.py                  # Automated training pipeline
├── Modelfile                 # Ollama model configuration
├── main.py                   # NiceGUI chatbot with Ollama
├── requirements.txt          # Python dependencies
└── SETUP.md                  # This file

Generated during training:
├── qwen3-nicegui-lora/       # LoRA adapter weights
├── qwen3-nicegui-merged/     # Merged model (HuggingFace format)
├── gguf_models/              # GGUF quantized models
└── llama.cpp/                # Conversion tools
```

## Expected Training Times

Hardware benchmarks (approximate):

| GPU | Training (3 epochs) | Merging | GGUF Conversion | Total |
|-----|---------------------|---------|-----------------|-------|
| RTX 4090 | 2-3 hours | 5 min | 15 min | ~3.5 hours |
| RTX 3090 | 4-6 hours | 8 min | 20 min | ~6.5 hours |
| A6000 | 3-4 hours | 6 min | 15 min | ~4.5 hours |

## Model Details

**Base Model:** Qwen2.5-Coder-24B-Instruct
- Developed by Alibaba Cloud
- 24 billion parameters
- Specialized for code generation
- Context length: 32K tokens

**Fine-tuning Method:** QLoRA (Quantized LoRA)
- 4-bit quantization for efficiency
- LoRA rank: 16
- Target modules: q_proj, k_proj, v_proj, o_proj
- Trainable parameters: ~0.2% of total

**Dataset:** ~300 NiceGUI examples
- Source: Official NiceGUI GitHub repository
- Format: Instruction-response pairs
- Coverage: UI components, layouts, events, advanced features

## Next Steps

After training, you can:

1. **Test extensively**: Try various NiceGUI questions
2. **Expand dataset**: Add more examples or your own code
3. **Retrain with adjustments**: Modify hyperparameters
4. **Share**: Export the model for others to use
5. **Integrate**: Use in your development workflow

## Support

For issues related to:
- **NiceGUI framework**: https://nicegui.io
- **Qwen models**: https://github.com/QwenLM/Qwen2.5
- **LoRA/PEFT**: https://github.com/huggingface/peft
- **Ollama**: https://ollama.ai

## License

This training pipeline uses:
- NiceGUI: MIT License
- Qwen2.5: Apache 2.0 License
- Transformers, PEFT: Apache 2.0 License

Your fine-tuned model inherits these licenses.


