import os
import sys
import json
import argparse
import wandb
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import ollama
from datetime import datetime

device = (
    torch.device("mps") if torch.backends.mps.is_available() # for M1 (Jacob)
    else torch.device("cuda") if torch.cuda.is_available()  # for GPU (Cole)
    else torch.device("cpu")
)
print(f"Using device: {device}")

# Ensure the scripts directory is in the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), "scripts"))

import scripts
from preprocess import load_and_preprocess
from train import train_model
from inference import generate_summaries
from evaluate_model import evaluate_model



shouldTrain = False
shouldUseHuggingFace = False
shouldJustEval = False # Jacob: skip evaluation on local run; Cole can set to True
shouldEvaluate = False  # Jacob: skip evaluation on local run; Cole can set to True
summary_method = "first_500" # "first_500" or "chunks_100", "chunks_100" is not fully developed yet

def main():
    """
    Runs the full pipeline for the BioLaySumm shared task:
    - Authenticates with Hugging Face and W&B
    - Loads and preprocesses datasets
    - Trains llama2
    - Runs inference to generate summaries
    - Evaluates generated summaries against gold summaries
    """

    # Step 1: Authenticate with Hugging Face
    if(shouldUseHuggingFace):
        print("\nLogging into Hugging Face...")
        login()

    #modelNames = ["llama2"] # not used for ollama
    models = ["llama2"]  # Using string identifiers for Ollama
    #tokenizers = [None]  # not used for ollama

    # Step 2: Authenticate with Weights & Biases
    print("\nStep 2: Logging into Weights & Biases...")
    wandb.login()

    api = wandb.Api()

    # Step 3: Initialize Weights & Biases logging
    print("\nStep 3: Inititalizing Weights & Biases...")
     # or get from args/config
    model_name = models[0]
    run_name = f"{model_name}_{summary_method}_{datetime.now().strftime('%m%d_%H%M')}"

    wandb.init(project="BioLaySumm-2025", name=run_name, mode="online")

    # Load model with HuggingFace (switched to Ollama)
    # modelNames = ["HuggingFaceTB/SmolLM-1.7B"]
    # tokenizers = []
    # models = []
    # smollm_tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-1.7B")
    # # Set a padding token if it's missing
    # if smollm_tokenizer.pad_token is None:
    #     smollm_tokenizer.pad_token = smollm_tokenizer.eos_token  # Use EOS as padding
    # smollm_model = AutoModelForCausalLM.from_pretrained(
    #     "HuggingFaceTB/SmolLM-1.7B", torch_dtype=torch.float16, device_map="auto"
    # )
    # tokenizers.append(smollm_tokenizer)
    # models.append(smollm_model)

    
    if not shouldJustEval:
        print("\nStep 4: Loading & Preprocessing Data...")
        train_data, dev_data = load_and_preprocess()

        print(type(dev_data))

        if(shouldTrain):
            print("\nStep 5: Training Models...")
            for model in models:
                train_model(train_data, model)
        else:
            print("\nStep 5: Skipping Training (shouldTrain is False)")

        print("\nStep 6: Running Inference...")
        # Load trained models
        i = 0
        for model in models:
            #modelNames = modelNames[i] # not used for ollama
            generate_summaries(model, dev_data, "llama2", summary_method) # "first_500" or "chunks_100", "chunks_100" is not fully developed yet
            i += 1

    if shouldEvaluate:
        model = models[0]
        print("\nStep 7: Evaluating Predictions...")
        eval_results = evaluate_model(model)
        wandb.log({f"{model}_metrics": eval_results})  # Log evaluation results
    else:
        print("\nStep 7: Skipping Evaluation (shouldEvaluate is False)")

    print("\nPipeline completed successfully!")

if __name__ == "__main__":
    main()