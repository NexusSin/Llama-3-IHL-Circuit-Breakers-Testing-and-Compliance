import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
ADAPTER_PATH = "/path/to/mistral_IHL_training/mistral_ihl_circuit_breaker_model"
OUTPUT_PATH = "/path/to/mistral_IHL_training/merged_mistral_ihl_model"

print("Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

print("Loading adapter...")
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)

print("Merging adapter with base model...")
merged_model = model.merge_and_unload()

print("Saving merged model...")
merged_model.save_pretrained(OUTPUT_PATH)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.save_pretrained(OUTPUT_PATH)

print(f"Merged model saved to {OUTPUT_PATH}")

