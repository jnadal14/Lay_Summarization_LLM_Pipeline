from datasets import load_dataset
from sklearn.model_selection import train_test_split

def validate_dataset(dataset, dataset_name):
    """
    Runs multiple validation checks to ensure the dataset is correctly loaded and structured.
    """
    assert isinstance(dataset, dict), f"ERROR: {dataset_name} is not a dictionary!"
    assert "train" in dataset, f"ERROR: {dataset_name} does not contain a 'train' split!"
    
    # Check first entry format
    first_entry = dataset["train"][0]
    assert "article" in first_entry, f"ERROR: {dataset_name} missing 'article' field!"
    assert "summary" in first_entry, f"ERROR: {dataset_name} missing 'summary' field!"
    
    print(f"{dataset_name} dataset loaded successfully with {len(dataset['train'])} samples.")

def load_and_preprocess():
    """
    Loads BioLaySumm datasets from Hugging Face, validates them, 
    and returns train/dev splits in-memory.
    """
    print("Loading datasets from Hugging Face...")
    PLOS = load_dataset("BioLaySumm/BioLaySumm2025-PLOS")
    eLife = load_dataset("BioLaySumm/BioLaySumm2025-eLife")

    # Validate datasets
    validate_dataset(PLOS, "PLOS")
    validate_dataset(eLife, "eLife")

    # Combine datasets
    full_data = list(PLOS["train"]) + list(eLife["train"])
    
    print(f"Total samples combined: {len(full_data)}")

    # Extract only required fields
    full_data = [dict(article=ex["article"], summary=ex["summary"]) for ex in full_data]

    # Split into train (90%) and dev (10%)
    train_data, dev_data = train_test_split(full_data, test_size=0.1, random_state=42)
    
    print(f"Data split: {len(train_data)} train, {len(dev_data)} dev samples.")
    
    return train_data, dev_data  # Returning instead of saving

if __name__ == "__main__":
    train, dev = load_and_preprocess()