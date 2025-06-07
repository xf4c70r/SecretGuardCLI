import os
import sys
import torch
from datasets import load_dataset
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model, PeftModel
from trl import SFTTrainer

# === CONFIGURATION ===
BASE_MODEL = "bigcode/starcoderbase"
DATASET_PATH = "Dataset/formatted_dataset_50k.jsonl"
OUTPUT_DIR = "output"
HF_REPO = "asudarshan/secretguard-starcoderbase-lora2"

# === TRAINING FUNCTION ===
def train():
    print("\n=== Training Mode ===\n")

    dataset = load_dataset("json", data_files=DATASET_PATH, split="train")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto"
    )

    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    peft_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["c_proj", "q_proj", "k_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    peft_model = get_peft_model(model, peft_config)

    training_arguments = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        learning_rate=2e-5,
        save_steps=1000,
        logging_steps=100,
        fp16=True,
        bf16=False,
        group_by_length=True,
        dataloader_num_workers=0,
        optim="adamw_torch",
        report_to=None,
    )

    def formatting_func(example):
        return example["text"]

    trainer = SFTTrainer(
        model=peft_model,
        train_dataset=dataset,
        peft_config=peft_config,
        formatting_func=formatting_func,
        args=training_arguments,
    )

    trainer.train()

    # Save PEFT model and tokenizer
    peft_model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"\nTraining complete. LoRA adapter and tokenizer saved to: {OUTPUT_DIR}")

    # Push to Hugging Face Hub
    print(f"\nPushing to Hugging Face Hub at: {HF_REPO}")
    peft_model.push_to_hub(HF_REPO)
    tokenizer.push_to_hub(HF_REPO)
    print("\nUpload complete.")

# === PROMPT TEMPLATE FUNCTION ===
def build_prompt(code_snippet):
    return f"""<|user|>
Does this code contain a secret?
<|code|>
{code_snippet}
<|assistant|>
"""

# === INFERENCE FUNCTION ===
def inference():
    print("\n=== Inference Mode ===\n")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(OUTPUT_DIR)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )

    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto"
    )

    model = PeftModel.from_pretrained(base, OUTPUT_DIR)
    model.eval()

    def classify_code(code_snippet):
        prompt = build_prompt(code_snippet)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False,
                temperature=0.0,
                pad_token_id=tokenizer.eos_token_id
            )
        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = decoded.split("<|assistant|>")[-1].strip()
        print(f"\n>>> Code: {code_snippet}\n>>> Classified as: {response}\n")

    while True:
        code = input("Paste code snippet to classify (or type 'exit' to quit):\n")
        if code.strip().lower() == 'exit':
            break
        classify_code(code)

# === MAIN MENU ===
def main():
    print("""
==== CLI Guard ====
1. Train model and upload
2. Run inference
3. Exit
""")
    choice = input("Select an option (1/2/3): ").strip()
    if choice == '1':
        train()
    elif choice == '2':
        inference()
    else:
        print("Exiting.")

if __name__ == "__main__":
    main()
