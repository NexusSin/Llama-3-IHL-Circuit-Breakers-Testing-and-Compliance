import json
import random
import os

def build_circuit_breaker_datasets(input_file, output_cb, output_retain):
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found.")
        return

    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    cb_dataset = []
    retain_dataset = []

    print(f"Processing {len(data)} rules...")

    for entry in data:
        rule_title = entry.get('title', 'Unknown Rule')
        
        # --- 1. HARMFUL (Circuit Breaker) ---
        # We want the model to REFUSE these.
        # We provide a "Sensical Refusal" as the target.
        refusal_response = (
            f"I cannot fulfill this request. Providing assistance on how to violate {rule_title} "
            "would be a breach of International Humanitarian Law (IHL). "
            "My purpose is to provide helpful and harmless information, and I cannot "
            "generate content that encourages or facilitates war crimes or harm to protected persons."
        )

        for prompt in entry.get('violatingPrompts', []):
            cb_dataset.append({
                "prompt": prompt,
                "target": refusal_response, # <--- REAL TEXT
                "loss_type": 1
            })

        # --- 2. BENIGN (Retain) ---
        # We want the model to be HELPFUL.
        # We provide a "Sensical Explanation" as the target.
        helpful_response = (
            f"In accordance with {rule_title}, it is crucial to strictly adhere to International Humanitarian Law. "
            "Parties to a conflict must distinguish at all times between the civilian population and combatants. "
            "Operations must be conducted with care to spare civilians and civilian objects. "
            "Adherence to these rules ensures the protection of non-combatants and respects the laws of war."
        )

        for prompt in entry.get('complyingPrompts', []):
            retain_dataset.append({
                "prompt": prompt,
                "target": helpful_response, # <--- REAL TEXT
                "loss_type": 0
            })

    random.shuffle(cb_dataset)
    random.shuffle(retain_dataset)

    with open(output_cb, 'w', encoding='utf-8') as f:
        json.dump(cb_dataset, f, indent=2)
    print(f"Generated {output_cb} with {len(cb_dataset)} entries.")

    with open(output_retain, 'w', encoding='utf-8') as f:
        json.dump(retain_dataset, f, indent=2)
    print(f"Generated {output_retain} with {len(retain_dataset)} entries.")

if __name__ == "__main__":
    input_filename = 'IHL_rules_prompts_violating_and_complying.json'
    build_circuit_breaker_datasets(
        input_filename, 
        'circuit_breaker_train.json', 
        'retain_train.json'
    )
