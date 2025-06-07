import json

def convert_to_instruction_format(example):
    code = example["code"].strip()
    label = example["label"].strip().upper()
    
    answer = "Yes" if label == "SECRET" else "No"
    
    instruction_text = (
        "<|user|>\n"
        "Does this code contain a secret?\n"
        "<|code|>\n"
        f"{code}\n"
        "<|assistant|>\n"
        f"{answer}"
    )
    
    return {"text": instruction_text}

input_path = "secret_detection_validation.jsonl"
output_path = "formatted_dataset_validation.jsonl"

with open(input_path, "r") as f:
    raw_data = [json.loads(line) for line in f]  # FIXED

formatted_data = [convert_to_instruction_format(entry) for entry in raw_data]

with open(output_path, "w") as f_out:
    for item in formatted_data:
        f_out.write(json.dumps(item) + "\n")
