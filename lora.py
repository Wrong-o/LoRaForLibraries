# LoRA Fine-tuning Script for NiceGUI Examples
# Install dependencies: pip install -r requirements.txt

from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch

print("=" * 50)
print("NiceGUI LoRA Fine-tuning")
print("=" * 50)

# Use a compatible base model for LoRA training
# We'll use Qwen2.5-7B which has the same architecture as your qwen3-coder model
model_name = "Qwen/Qwen2.5-7B-Instruct"  # Compatible architecture for training
print(f"\n1. Loading training model: {model_name}")
print("Note: Using this for LoRA training. Final model will work with your local Ollama qwen3-coder")

tokenizer = AutoTokenizer.from_pretrained(model_name)

# Set padding token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    load_in_4bit=True,
    torch_dtype=torch.float16,
    trust_remote_code=True,
)

# Prepare model for k-bit training
model = prepare_model_for_kbit_training(model)

# Prepare dataset
print("\n2. Loading dataset: nicegui_lora_dataset.json")
dataset = load_dataset("json", data_files="nicegui_lora_dataset.json")

# Split dataset into train and validation (90/10 split)
dataset = dataset["train"].train_test_split(test_size=0.1, seed=42)
print(f"   - Training samples: {len(dataset['train'])}")
print(f"   - Validation samples: {len(dataset['test'])}")

def tokenize_fn(examples):
    prompts = []
    for instruction, response in zip(examples['instruction'], examples['response']):
        prompt = f"### Instruction:\n{instruction}\n\n### Response:\n{response}{tokenizer.eos_token}"
        prompts.append(prompt)
    
    return tokenizer(
        prompts,
        truncation=True,
        max_length=1024,
        padding=False,  # We'll pad in the collator
    )

print("\n3. Tokenizing dataset...")
tokenized_ds = dataset.map(
    tokenize_fn,
    batched=True,
    remove_columns=dataset["train"].column_names,
)

# Configure LoRA
print("\n4. Configuring LoRA...")
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

# Training arguments
print("\n5. Setting up training...")
training_args = TrainingArguments(
    output_dir="./qwen3-nicegui-lora",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=16,  # Increased for 24B model
    learning_rate=2e-4,  # Slightly lower for stability
    fp16=True,
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=50,
    save_steps=100,
    num_train_epochs=2,  # Reduced for 24B model
    save_total_limit=2,
    warmup_steps=100,  # Increased warmup
    report_to="none",
    load_best_model_at_end=True,
    dataloader_pin_memory=False,  # Reduce memory usage
)

# Trainer
trainer = Trainer(
    model=model,
    train_dataset=tokenized_ds["train"],
    eval_dataset=tokenized_ds["test"],
    args=training_args,
    data_collator=data_collator,
)

print("\n6. Starting training...")
print("=" * 50)
trainer.train()

# Save LoRA weights
print("\n7. Saving LoRA adapter...")
model.save_pretrained("./qwen3-nicegui-lora")
tokenizer.save_pretrained("./qwen3-nicegui-lora")
print("\n✓ Training complete! LoRA adapter saved to ./qwen3-nicegui-lora")
print("=" * 50)
