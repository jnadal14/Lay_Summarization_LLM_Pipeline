import json
import os
import torch

def save_json(data, filename):
    """
    Saves data to a JSON file.
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)

def load_json(filename):
    """
    Loads data from a JSON file.
    """
    if os.path.exists(filename):
        with open(filename, "r") as f:
            return json.load(f)
    return None

def save_model(model, model_path):
    """
    Saves a trained model to disk.
    """
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

def load_model(model, model_path):
    """
    Loads a trained model from disk.
    """
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        print(f"Model loaded from {model_path}")
        return model
    else:
        print(f"Model not found at {model_path}")
        return None