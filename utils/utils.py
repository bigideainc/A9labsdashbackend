import pandas as pd
from datasets import Dataset, DatasetDict
import re


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




