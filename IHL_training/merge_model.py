import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
ADAPTER_PATH = "/home/users/fdollaku/circuit-breakers/IHL_training/ihl_circuit_breaker_model"
MERGED_OUTPUT_PATH = "/home/users/fdollaku/circuit-breakers/IHL_training/ihl_llama3_merged"

print("Loading and Merging...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    torch_dtype=torch.float16,
    device_map="auto"
)

model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
model = model.merge_and_unload() # <--- This fuses the weights permanently

print(f"Saving merged model to {MERGED_OUTPUT_PATH}...")
model.save_pretrained(MERGED_OUTPUT_PATH)
tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)
tokenizer.save_pretrained(MERGED_OUTPUT_PATH)
print("Done.")
