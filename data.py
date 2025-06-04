import os
from datasets import load_dataset
import json

def prepare_sst2_dataset(output_base_path="datasets/clean"):
    """
    Downloads the SST-2 dataset, and saves its splits (train, validation, test)
    as JSON files in the specified directory structure.

    Args:
        output_base_path (str): The base directory to save the processed datasets.
                                Default is 'datasets/clean'.
    """
    dataset_name = "sst2"
    output_dir = os.path.join(output_base_path, dataset_name)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading {dataset_name} dataset...")
    # SST-2 typically has 'train', 'validation', 'test' splits.
    # load_dataset("sst2") will return a DatasetDict with these keys.
    sst2_dataset_dict = load_dataset(dataset_name)

    split_mapping = {
        "train": "train.json",
        "validation": "dev.json",  # Mapping validation to dev.json as requested
        "test": "test.json"
    }

    for split_name, original_split_data in sst2_dataset_dict.items():
        if split_name in split_mapping:
            output_file_name = split_mapping[split_name]
            output_file_path = os.path.join(output_dir, output_file_name)
            
            print(f"Processing split: {split_name} -> {output_file_name}")
            # Convert dataset to list of dicts for json.dump for more control if needed,
            # or use dataset.to_json() directly if available and suitable.
            # Using to_json for simplicity as it's a direct feature of Hugging Face datasets.
            try:
                original_split_data.to_json(output_file_path)
                print(f"Saved {split_name} split to {output_file_path}")
            except Exception as e:
                print(f"Error saving {split_name} to JSON: {e}")

        else:
            print(f"Skipping unmapped split: {split_name}")

    print(f"SST-2 dataset preparation complete. Files saved in {output_dir}")

if __name__ == "__main__":
    prepare_sst2_dataset()
    print("Check the datasets/clean/sst2/ directory for train.json, dev.json, and test.json")
