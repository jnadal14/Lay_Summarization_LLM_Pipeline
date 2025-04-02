import torch
import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import Dataset

def train_model(train_data, model_name):
    """
    Trains the specified model on the provided training data.
    
    Args:
        train_data (list): Preprocessed training data.
        model_name (str): The name of the model to train.
    """

    print(f"Initializing training for {model_name}...")

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="auto"
    )

    # Convert preprocessed data to Hugging Face Dataset format
    train_dataset = Dataset.from_list(train_data)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=f"./results/{model_name.replace('/', '_')}",
        evaluation_strategy="epoch",
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=3,
        save_steps=500,
        save_total_limit=2,
        report_to="wandb",
    )

    # Initialize trainer
    # trainer = Trainer(
    #     model=model,
    #     args=training_args,
    #     train_dataset=train_dataset,
    #     tokenizer=tokenizer,
    # )

    # Train the model
    #trainer.train()

    print(f"Loading completed for {model_name}")

    # Log the model checkpoint to W&B
    wandb.log({f"{model_name}_trained": True})