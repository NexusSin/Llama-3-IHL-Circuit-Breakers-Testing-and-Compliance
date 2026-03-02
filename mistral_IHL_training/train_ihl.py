import os
import json
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset, concatenate_datasets

# ==========================================
#              CONFIGURATION
# ==========================================

DATA_DIR = "/mnt/aiongpfs/users/fdollaku/circuit-breakers/mistral_IHL_training/building_dataset"
OUTPUT_DIR = "/mnt/aiongpfs/users/fdollaku/circuit-breakers/mistral_IHL_training/mistral_ihl_circuit_breaker_model"

CB_DATASET_PATH = os.path.join(DATA_DIR, "circuit_breaker_train.json")
RETAIN_DATASET_PATH = os.path.join(DATA_DIR, "retain_train.json")

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"

# HYPERPARAMETERS
LORRA_ALPHA = 0.5
TARGET_LAYERS = [12, 16, 20, 24, 28]
LORA_R = 8
LORA_DROPOUT = 0.05
LEARNING_RATE = 2e-4

BATCH_SIZE = 1
GRAD_ACCUMULATION = 8
NUM_EPOCHS = 1
MAX_LENGTH = 512

# ==========================================
#           DATA LOADING LOGIC
# ==========================================

def load_and_tokenize_data(tokenizer):
    if not os.path.exists(CB_DATASET_PATH):
        raise FileNotFoundError(f"Error: Could not find {CB_DATASET_PATH}")
    if not os.path.exists(RETAIN_DATASET_PATH):
        raise FileNotFoundError(f"Error: Could not find {RETAIN_DATASET_PATH}")

    def format_and_tokenize(examples, is_harmful: bool):
        prompts = examples["prompt"]
        targets = examples["target"]

        input_ids_list = []
        labels_list = []
        loss_type_list = []

        for prompt, target in zip(prompts, targets):
            messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": target},
            ]

            enc = tokenizer.apply_chat_template(
                messages,
                truncation=True,
                max_length=MAX_LENGTH,
                padding="max_length",
                return_tensors="pt",
                tokenize=True,
            )

            input_ids = enc[0]
            labels = input_ids.clone()

            input_ids_list.append(input_ids)
            labels_list.append(labels)
            loss_type_list.append(1 if is_harmful else 0)

        return {
            "input_ids": input_ids_list,
            "labels": labels_list,
            "loss_type": loss_type_list,
        }

    print(f"Loading harmful data from {CB_DATASET_PATH} (circuit breaker / refusal)...")
    with open(CB_DATASET_PATH, "r") as f:
        cb_data = json.load(f)
    cb_dataset = Dataset.from_list(cb_data)
    cb_dataset = cb_dataset.map(
        lambda x: format_and_tokenize(x, is_harmful=True),
        batched=True,
        remove_columns=cb_dataset.column_names,
    )

    print(f"Loading benign data from {RETAIN_DATASET_PATH} (retain / comply)...")
    with open(RETAIN_DATASET_PATH, "r") as f:
        retain_data = json.load(f)
    retain_dataset = Dataset.from_list(retain_data)
    retain_dataset = retain_dataset.map(
        lambda x: format_and_tokenize(x, is_harmful=False),
        batched=True,
        remove_columns=retain_dataset.column_names,
    )

    return cb_dataset, retain_dataset

# ==========================================
#           CUSTOM TRAINER
# ==========================================

class CircuitBreakerTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_fct = torch.nn.CrossEntropyLoss()

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        loss_type = inputs.pop("loss_type", None)
        outputs = model(**inputs, output_hidden_states=True)

        # 1. Generation Loss
        logits = outputs.get("logits")
        labels = inputs.get("labels")
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        gen_loss = self.loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )

        # 2. Circuit Breaker Loss (original logic)
        hidden_states = outputs.hidden_states
        selected_hidden_states = [hidden_states[i] for i in TARGET_LAYERS]

        cb_loss = torch.tensor(0.0, device=model.device)

        if loss_type is not None:
            mask = loss_type.view(-1, 1, 1).to(model.device)
            num_harmful = mask.sum()
            if num_harmful > 0:
                for layer_tensor in selected_hidden_states:
                    masked_tensor = layer_tensor * mask
                    normalization_factor = (
                        num_harmful * layer_tensor.size(1) * layer_tensor.size(2)
                    )
                    cb_loss += torch.norm(masked_tensor, p=2) / normalization_factor
        else:
            for layer_tensor in selected_hidden_states:
                cb_loss += torch.norm(layer_tensor, p=2) / layer_tensor.numel()

        total_loss = gen_loss + (LORRA_ALPHA * cb_loss)
        return (total_loss, outputs) if return_outputs else total_loss

# ==========================================
#           MAIN EXECUTION
# ==========================================

def main():
    # single-GPU behavior matching original setup[cite:2]
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    torch.cuda.empty_cache()
    print(f"Loading Base Model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading model to GPU 0...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map={"": 0},
        dtype=torch.bfloat16,
    )

    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=LORA_R,
        lora_alpha=32,
        lora_dropout=LORA_DROPOUT,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    cb_dataset, retain_dataset = load_and_tokenize_data(tokenizer)
    train_dataset = concatenate_datasets([cb_dataset, retain_dataset])
    print(f"Total training examples: {len(train_dataset)}")

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUMULATION,
        learning_rate=LEARNING_RATE,
        num_train_epochs=NUM_EPOCHS,
        save_strategy="epoch",
        fp16=False,
        bf16=True,
        remove_unused_columns=False,
        report_to="none",
        gradient_checkpointing=True,
        deepspeed=None,
        # suppress logs / tqdm
        logging_strategy="no",
        disable_tqdm=True,
    )

    trainer = CircuitBreakerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer, padding=True, pad_to_multiple_of=8
        ),
    )

    print("Starting Training...")
    trainer.train()

    print(f"Training finished. Saving model to: {OUTPUT_DIR}")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)


if __name__ == "__main__":
    main()

