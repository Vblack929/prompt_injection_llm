import json
import jsonlines
import random
import sys
import os

# Add the parent directory to the path so we can import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TEST_INJECTED_WORD, TEST_INJECTED_PROMPT, IGNORE_ATTACK_SENTENCES

def load_jsonl(path):
    with open(path, 'r') as f:
        return [json.loads(line) for line in f]

def generate_format_dpo_data(data_path, prompt_inject, target_output, output_path, num_samples):
    data = load_jsonl(data_path)
    dpo_data = []

    for d in data[:num_samples]:
        injected_prompt = d['instruction'] + "\n\n" + d['input'] + "\n\n" + prompt_inject

        dpo_entry = {
            "prompt": injected_prompt,
            "chosen": f"[SYSTEM] Injection detected and blocked. [/SYSTEM]\n[USER] {d['output']}",
            "rejected": target_output
        }
        dpo_data.append(dpo_entry)

    # Write to jsonl file
    with jsonlines.open(output_path, mode='w') as writer:
        writer.write_all(dpo_data)

    print(f"Generated {len(dpo_data)} DPO samples saved to {output_path}")

def generate_random_dpo_data(data_path, output_path, num_samples):
    data = load_jsonl(data_path)
    dpo_data = []
    num_samples = min(num_samples, len(data))
    for d in data[:num_samples]:
        attack_sentence = random.choice(IGNORE_ATTACK_SENTENCES['train'])
        formated_attack = attack_sentence.format(injected_prompt=TEST_INJECTED_PROMPT)
        prompt = d['instruction'] + "\n\n" + d['input']
        injected_prompt = prompt + "\n\n" + formated_attack
        dpo_entry = {
            "prompt": injected_prompt,
            "chosen": d['output'],
            "rejected": TEST_INJECTED_WORD
        }
        dpo_data.append(dpo_entry)

    with jsonlines.open(output_path, mode='w') as writer:
        writer.write_all(dpo_data)

    print(f"Generated {len(dpo_data)} DPO samples saved to {output_path}")
    
if __name__ == "__main__":
    data_path = "datasets/alpaca_data_with_input_train.jsonl"
    output_path = "dpo/data/random_dpo_data_train_800.jsonl"
    num_samples = 800
    generate_random_dpo_data(data_path, output_path, num_samples)
            