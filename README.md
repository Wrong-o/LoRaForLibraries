# NiceGUI LoRA Fine-tuning Project

Fine-tune a large language model to become an expert NiceGUI coding assistant.

## What This Does

This project trains an LLM (Qwen2.5-Coder-24B) on NiceGUI examples to create a specialized coding assistant that understands NiceGUI deeply. The fine-tuned model runs locally using Ollama and integrates with a NiceGUI chatbot interface.

## Quick Start

See [QUICKSTART.md](QUICKSTART.md) for a 5-minute overview.

## Full Documentation

See [SETUP.md](SETUP.md) for complete setup instructions, troubleshooting, and technical details.

## Components

### 1. Data Collection
- `download_examples.py` - Scrapes NiceGUI examples from GitHub
- `nicegui_lora_dataset.json` - ~300 instruction-response pairs

### 2. Training Pipeline
- `train.py` - Automated training orchestrator
- `lora.py` - LoRA fine-tuning script (QLoRA with 4-bit quantization)
- `requirements.txt` - Python dependencies

### 3. Model Conversion
- `merge_lora.py` - Merge LoRA adapter with base model
- `convert_to_gguf.sh` - Convert to GGUF format for Ollama
- `Modelfile` - Ollama model configuration

### 4. Deployment
- `main.py` - NiceGUI chatbot with Ollama backend
- Includes function calling (time, calculator)
- Multi-chat support with history

## Requirements

- **GPU**: 24GB+ VRAM (RTX 4090, RTX 3090, A6000)
- **RAM**: 32GB+ recommended
- **Storage**: ~50GB free space
- **Time**: 3-7 hours total (mostly training)

## Usage Workflow

```
Download Examples → Train LoRA → Merge Model → Convert GGUF → Ollama → Chatbot
```

1. **Train**: `python train.py` (2-4 hours)
2. **Merge**: `python merge_lora.py` (5 mins)
3. **Convert**: `bash convert_to_gguf.sh` (15 mins)
4. **Deploy**: `ollama create qwen3-nicegui -f Modelfile`
5. **Use**: `python main.py` or `ollama run qwen3-nicegui`

## Example Queries

Try asking the fine-tuned model:

- "How do I create a button in NiceGUI?"
- "Show me how to build a data table with NiceGUI"
- "How do I add authentication to a NiceGUI app?"
- "Create a chat interface using NiceGUI"

## Architecture

**Base Model**: Qwen2.5-Coder-24B-Instruct (24B parameters)  
**Fine-tuning**: QLoRA (4-bit quantization, rank 16)  
**Trainable**: ~40M parameters (0.2% of total)  
**Output**: Q4_K_M GGUF (~13GB)  
**Inference**: Ollama (local, GPU-accelerated)

## Training Details

- **Method**: Low-Rank Adaptation (LoRA) with 4-bit quantization
- **Epochs**: 2
- **Batch Size**: 1 x 8 (gradient accumulation)
- **Learning Rate**: 3e-4
- **LoRA Rank**: 16
- **Target Modules**: q_proj, k_proj, v_proj, o_proj

## Performance

The fine-tuned model should:
- Generate accurate NiceGUI code
- Follow NiceGUI conventions and best practices
- Provide working examples with proper imports
- Understand NiceGUI-specific concepts (elements, events, bindings)

## Project Structure

```
nicegui/
├── README.md                 # This file
├── QUICKSTART.md             # Quick start guide
├── SETUP.md                  # Detailed setup instructions
├── requirements.txt          # Python dependencies
│
├── download_examples.py      # Data collection
├── nicegui_lora_dataset.json # Training data
│
├── train.py                  # Training orchestrator
├── lora.py                   # LoRA training
├── merge_lora.py             # Model merging
├── convert_to_gguf.sh        # GGUF conversion
├── Modelfile                 # Ollama config
│
└── main.py                   # Chatbot interface
```

## Contributing

To improve the model:

1. **Add more examples**: Edit `nicegui_lora_dataset.json`
2. **Adjust hyperparameters**: Modify `lora.py`
3. **Retrain**: Run `python train.py` again
4. **Test**: Use `ollama run qwen3-nicegui`

## License

- Training scripts: MIT
- NiceGUI: MIT
- Qwen2.5: Apache 2.0
- Fine-tuned model: Inherits licenses

## Resources

- [NiceGUI Documentation](https://nicegui.io)
- [Qwen2.5 Model](https://github.com/QwenLM/Qwen2.5)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [Ollama](https://ollama.ai)

---

**Status**: Ready to train ✓  
**Next Step**: Run `python train.py`


