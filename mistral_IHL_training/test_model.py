import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# --- CONFIGURATION ---
BASE_MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"
ADAPTER_PATH = "/mnt/aiongpfs/users/fdollaku/circuit-breakers/mistral_IHL_training/mistral_ihl_circuit_breaker_model"

def main():
    print("1. Loading Base Model...")
    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_ID,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
    except Exception as e:
        print(f"Error loading base model: {e}")
        return

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print("2. Loading Circuit Breaker Adapter...")
    try:
        model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    except Exception as e:
        print(f"\nERROR: Could not load adapters from {ADAPTER_PATH}")
        print(f"Details: {e}")
        print("Did the training finish successfully and OUTPUT_DIR match ADAPTER_PATH?")
        return

    model.eval()
    print("\nModel Loaded Successfully! Let's test it.\n")

    print("-" * 50)
    print("MISTRAL IHL CIRCUIT BREAKER TEST CONSOLE")
    print("-" * 50)

    while True:
        user_input = input("\nEnter a prompt (or 'q' to quit): ")
        if user_input.lower() == "q":
            break

        messages = [
            {"role": "user", "content": user_input},
        ]

        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        ).to(model.device)

        print("Generating response...")
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                pad_token_id=tokenizer.pad_token_id,
                max_new_tokens=256,
                eos_token_id=tokenizer.eos_token_id,
                do_sample=True,
                temperature=0.6,
                top_p=0.9,
            )

        response = outputs[0][inputs.input_ids.shape[-1]:]
        print("\n--- RESPONSE ---")
        print(tokenizer.decode(response, skip_special_tokens=True))
        print("----------------")


if __name__ == "__main__":
    main()

