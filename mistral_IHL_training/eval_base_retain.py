# eval_base_retain.py

import os
import json
import csv
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

BASE_MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"

RETAIN_PATH = "/mnt/aiongpfs/users/fdollaku/circuit-breakers/mistral_IHL_training/building_dataset/retain_train.json"
OUT_CSV = "/mnt/aiongpfs/users/fdollaku/circuit-breakers/mistral_IHL_training/checking_ihl_compatability/mistral_base_retain_eval.csv"

MAX_NEW_TOKENS = 256
TEMPERATURE = 0.6
TOP_P = 0.9


def load_retain_dataset(path):
    with open(path, "r") as f:
        data = json.load(f)
    return data  # list of {"prompt": ..., "target": ...}


def load_base_model():
    print("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer, model


def generate(model, tokenizer, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            pad_token_id=tokenizer.pad_token_id,
            max_new_tokens=MAX_NEW_TOKENS,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=True,
            temperature=TEMPERATURE,
            top_p=TOP_P,
        )

    response = outputs[0][inputs.input_ids.shape[-1]:]
    return tokenizer.decode(response, skip_special_tokens=True)


def main():
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)

    retain_data = load_retain_dataset(RETAIN_PATH)
    tokenizer, base_model = load_base_model()

    print(f"Loaded {len(retain_data)} retain prompts for BASE model.")

    with open(OUT_CSV, "w", newline="") as f_out:
        fieldnames = ["id", "prompt", "target", "base_response"]
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()

        for i, ex in enumerate(retain_data):
            prompt = ex["prompt"]
            target = ex.get("target", "")

            base_resp = generate(base_model, tokenizer, prompt)

            writer.writerow(
                {
                    "id": i,
                    "prompt": prompt,
                    "target": target,
                    "base_response": base_resp,
                }
            )

            if (i + 1) % 20 == 0:
                print(f"[BASE] Evaluated {i+1}/{len(retain_data)} prompts...")

    print(f"BASE results saved to: {OUT_CSV}")


if __name__ == "__main__":
    main()

