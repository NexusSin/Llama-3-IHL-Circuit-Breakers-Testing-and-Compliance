# ihl_analyze.py
import csv

CSV_PATH = "/home/users/fdollaku/circuit-breakers/out/ihl_eval_outputs.csv"

violating = []
complying = []

with open(CSV_PATH, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row["prompt_type"] == "violating":
            violating.append(row)
        else:
            complying.append(row)

print(f"Violating prompts: {len(violating)}")
print(f"Complying prompts: {len(complying)}")

# Show a few example answers for violating prompts
print("\nSample violating prompts and answers:\n")
for row in violating[:5]:
    print(f"Rule {row['ruleId']} - {row['title']}")
    print("PROMPT :", row["prompt"])
    print("ANSWER :", row["model_answer"][:500].replace("\n", " "))
    print("-" * 80)

