import json
import csv
from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

MODEL_PATH = "/home/users/fdollaku/circuit-breakers/out/Llama-3-8b_CB"
IHL_JSON = "/home/users/fdollaku/circuit-breakers/out/IHL_rules_prompts_violating_and_complying.json"
OUT_CSV = "/home/users/fdollaku/circuit-breakers/out/ihl_eval_outputs.csv"

def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        fix_mistral_regex=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        dtype=torch.float16 if device == "cuda" else torch.float32,
    )
    model.to(device)
    model.eval()
    return tokenizer, model, device

def generate_answer(tokenizer, model, device, prompt, max_new_tokens=96):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.3,
            top_p=0.8,
            repetition_penalty=1.1,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    text = tokenizer.decode(output[0], skip_special_tokens=True).strip()
    if text.startswith(prompt):
        text = text[len(prompt):].strip()
    return text

def main():
    tokenizer, model, device = load_model()

    data = json.loads(Path(IHL_JSON).read_text())

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "ruleId",
            "title",
            "prompt_type",   # "violating" or "complying"
            "prompt",
            "model_answer",
        ])

        for entry in data:
            rule_id = entry["ruleId"]
            title = entry["title"]

            for p in entry.get("violatingPrompts", []):
                ans = generate_answer(tokenizer, model, device, p)
                writer.writerow([rule_id, title, "violating", p, ans])

            for p in entry.get("complyingPrompts", []):
                ans = generate_answer(tokenizer, model, device, p)
                writer.writerow([rule_id, title, "complying", p, ans])

    print(f"Saved outputs to {OUT_CSV}")

if __name__ == "__main__":
    main()

