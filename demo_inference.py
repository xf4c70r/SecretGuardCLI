import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

HF_REPO = "asudarshan/secretguard-starcoderbase-lora2"
BASE_MODEL = "bigcode/starcoderbase"

# Setup quantization (if using LoRA with 4-bit)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto"
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(HF_REPO)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Load LoRA adapter
model = PeftModel.from_pretrained(base_model, HF_REPO)
model.eval()

def build_prompt(code_snippet):
    return f"""<|user|>
Does this code contain a secret?
<|code|>
{code_snippet}
<|assistant|>
"""

def classify(code_snippet):
    prompt = build_prompt(code_snippet)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=10,
            do_sample=False,
            temperature=0.0,
            pad_token_id=tokenizer.eos_token_id
        )
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return decoded.split("<|assistant|>")[-1].strip()

# Example
code = 'const password = "AKIA1234567890EXAMPLE";'
result = classify(code)
print(f"Classified as: {result}")