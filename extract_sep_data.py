import json
import pandas as pd
import os

def load_sep_dataset(file_path="Should-It-Be-Executed-Or-Processed/SEP_dataset/SEP_dataset.json"):
    """
    Load the SEP dataset from JSON file.
    
    Returns:
        list: List of SEP examples, each containing:
            - system_prompt_clean: Clean instruction
            - prompt_clean: Clean data  
            - prompt_instructed: Data with probe
            - system_prompt_instructed: Instruction with probe
            - witness: Expected probe answer
            - info: Metadata (task type, subtask, etc.)
    """
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def extract_clean_examples(sep_data):
    """Extract clean examples (instruction + data without probes)."""
    clean_examples = []
    for item in sep_data:
        clean_examples.append({
            'instruction': item['system_prompt_clean'],
            'data': item['prompt_clean'],
            'task_type': item['info']['type'],
            'subtask': item['info']['subtask']
        })
    return clean_examples

def extract_probe_examples(sep_data):
    """Extract examples with probes for testing instruction-data separation."""
    probe_examples = []
    for item in sep_data:
        # Probe in data (should NOT execute)
        probe_examples.append({
            'instruction': item['system_prompt_clean'],
            'data': item['prompt_instructed'],  # Data with probe
            'probe_location': 'data',
            'witness': item['witness'],
            'task_type': item['info']['type'],
            'is_insistent': item['info']['is_insistent']
        })
        
        # Probe in instruction (should execute)
        probe_examples.append({
            'instruction': item['system_prompt_instructed'],  # Instruction with probe
            'data': item['prompt_clean'],
            'probe_location': 'instruction', 
            'witness': item['witness'],
            'task_type': item['info']['type'],
            'is_insistent': item['info']['is_insistent']
        })
    return probe_examples

def save_as_csv(data, filename):
    """Save data as CSV file."""
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    print(f"Saved {len(data)} examples to {filename}")

def save_as_json(data, filename):
    """Save data as JSON file."""
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Saved {len(data)} examples to {filename}")

# Example usage
if __name__ == "__main__":
    # Load the full SEP dataset
    sep_data = load_sep_dataset()
    output_dir = "datasets/sep_dataset"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Loaded {len(sep_data)} SEP examples")
    
    # Extract clean examples (no probes)
    clean_examples = extract_clean_examples(sep_data)
    print(f"Extracted {len(clean_examples)} clean examples")
    
    # Extract probe examples for testing
    probe_examples = extract_probe_examples(sep_data)
    print(f"Extracted {len(probe_examples)} probe examples")
    
    # Save in different formats
    save_as_json(clean_examples, f"{output_dir}/sep_clean_examples.json")
    save_as_json(probe_examples, f"{output_dir}/sep_probe_examples.json")
    save_as_csv(clean_examples, f"{output_dir}/sep_clean_examples.csv")
    save_as_csv(probe_examples, f"{output_dir}/sep_probe_examples.csv")
    
    # Show sample data
    print("\nSample clean example:")
    print(json.dumps(clean_examples[0], indent=2))
    
    print("\nSample probe example:")
    print(json.dumps(probe_examples[0], indent=2)) 