import pandas as pd
from datasets import Dataset, DatasetDict
import math

def transform_csv_to_hf_dataset(csv_path, repo_name, seed=42, split_ratio=0.2):
    """
    Load a CSV file with 'instruction' and 'response' columns, transform it into the 
    'guanaco' template format, and upload it to Hugging Face.

    Parameters:
        csv_path (str): Path to the input CSV file.
        repo_name (str): The name of the Hugging Face dataset repository (e.g., 'username/repo-name').
        seed (int): Random seed for shuffling the data.
        split_ratio (float): Proportion of the dataset to use for testing.

    Returns:
        str: The Hugging Face dataset repository name.
    """
    # Load the CSV file into a pandas DataFrame
    df = pd.read_csv(csv_path)

    # Ensure the DataFrame has the expected columns
    if 'instruction' not in df.columns or 'response' not in df.columns:
        raise ValueError("CSV must contain 'instruction' and 'response' columns.")

    # Reformat each row using the guanaco template
    def transform_row(row):
        instruction = row['instruction'].strip()
        response = row['response'].strip()
        return f'<s>[INST] {instruction} [/INST] {response} </s>'

    # Apply the transformation to each row
    df['text'] = df.apply(transform_row, axis=1)

    # Drop the original columns, keeping only the reformatted 'text'
    df = df[['text']]

    # Create a Hugging Face Dataset from the DataFrame
    hf_dataset = Dataset.from_pandas(df)

    # Shuffle and split the dataset into train and test sets
    train_test_split = hf_dataset.shuffle(seed=seed).train_test_split(test_size=split_ratio)
    hf_dataset_dict = DatasetDict({
        'train': train_test_split['train'],
        'test': train_test_split['test']
    })

    # Push the dataset to the Hugging Face Hub
    hf_dataset_dict.push_to_hub(repo_name)

    # Return the repository name
    return repo_name

def calculate_system_requirements(dataset_size_mb):
    """
    Calculate minimum and recommended system specifications based on dataset size.
    """
    # Constants for the heuristic calculation (these values can be tuned based on empirical data)
    gpu_scaling_factor = 0.1  # GPU in TFLOPS per GB of dataset
    vram_scaling_factor = 0.5  # VRAM in GB per GB of dataset
    ram_scaling_factor = 1.0  # RAM in GB per GB of dataset
    storage_scaling_factor = 2.0  # Storage in GB per GB of dataset
    network_scaling_factor = 10.0  # Network in Mbps per GB of dataset

    # Minimum requirements
    minimum_specs = {
        "GPU": f"{max(1, math.ceil(gpu_scaling_factor * dataset_size_mb / 1024))} TFLOPS",
        "VRAM": f"{max(1, math.ceil(vram_scaling_factor * dataset_size_mb / 1024))} GB",
        "RAM": f"{max(2, math.ceil(ram_scaling_factor * dataset_size_mb / 1024))} GB",
        "Storage": f"{math.ceil(storage_scaling_factor * dataset_size_mb / 1024)} GB",
        "Network": f"{math.ceil(network_scaling_factor * dataset_size_mb)} Mbps",
    }

    # Recommended requirements
    recommended_specs = {
        "GPU": f"{max(4, math.ceil(2 * gpu_scaling_factor * dataset_size_mb / 1024))} TFLOPS",
        "VRAM": f"{max(8, math.ceil(2 * vram_scaling_factor * dataset_size_mb / 1024))} GB",
        "RAM": f"{max(8, math.ceil(2 * ram_scaling_factor * dataset_size_mb / 1024))} GB",
        "Storage": f"{math.ceil(3 * storage_scaling_factor * dataset_size_mb / 1024)} GB",
        "Network": f"{math.ceil(2 * network_scaling_factor * dataset_size_mb)} Mbps",
    }

    return {"minimum": minimum_specs, "recommended": recommended_specs}






