import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import random 
import string
import json
import time
from preprocess import load_and_preprocess
import ollama
from pathlib import Path
import wandb

# def generate_summary(model, tokenizer, text):
#     text = "Please summarize the following: " + text 
#     inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
#     input_ids = inputs.input_ids.to("cuda")
#     attention_mask = inputs.attention_mask.to("cuda")
#     tempInputText = tokenizer.decode(input_ids[0], skip_special_tokens=True)
#     true_start_time = time.time()
#     output_ids = model.generate(input_ids, attention_mask = attention_mask, max_length=612, do_sample=True, temperature=0.7, top_p=0.9)
#     print("time to compute output:", time.time() - true_start_time)
#     return (tokenizer.decode(output_ids[0], skip_special_tokens=True), tempInputText)

def truncate_to_words(text, max_words=500):
    return " ".join(text.split()[:max_words])

# new function using Ollama
def generate_summary_first_500(model_name, text):
    truncated_text = truncate_to_words(text, max_words=500)
    prompt = f"""
    You are a summarization assistant for medical scientific articles. Please rewrite the excerpt below in under 100 words, using everyday language. Focus on the main findings and their importance.

    IMPORTANT RULES:
    1) Output ONLY a single JSON list with one string, and nothing else.
    2) No disclaimers, no 'Here is a summary...', no self-references.
    3) Example of correct output format:
    ["This study found that..."]

    Text:
    {truncated_text}
    """.strip()
    true_start_time = time.time()
    response = ollama.chat(
        model=model_name,
        messages=[{'role': 'user', 'content': prompt}]
    )
    print("time to compute output:", time.time() - true_start_time)
    return (response['message']['content'], truncated_text)


def generate_summary_chunks_100(model_name, text):
    text_list = text.split()
    chunked_summaries = []
    true_start_time = time.time()
    for i in range(0, len(text_list), 100):
        chunk_start_time = time.time()
        chunk = text_list[i:i+100]
        chunk_str = " ".join(chunk)
        print(chunk_str)
        print(f"chunk length: {len(chunk_str)}")
        print("_______________________")
        prompt = f"Summarize the following truncated portion of a text in plain language:\n\n{chunk_str}"
        chunk_response = ollama.chat(
            model=model_name,
            messages=[{'role': 'user', 'content': prompt}]
        )['message']['content']
        chunked_summaries.append(chunk_response)
        print("time to ouput 1 chunked summary:", time.time() - chunk_start_time)

    print("time to ouput 1 full summary:", time.time() - true_start_time)
    final_response = " ".join(chunked_summaries)
    return (final_response, text)
    
    
# old function using HuggingFace
# def generate_summaries(tokenizer, model, dev_data, model_name):
#     # Generate summaries for a subset
#     num_samples = 20

#     print("predicting with llama2")
#     predictions_smollm_ini = [generate_summary(model_name, ex["article"]) for ex in dev_data[:num_samples]]
#     predictions_smollm = []
#     inputTexts = []
#     i = 0
#     while(i < len(predictions_smollm_ini)):
#         tempPred = predictions_smollm_ini[i][0]
#         tempInput = predictions_smollm_ini[i][1]
#         predictions_smollm.append(tempPred)
#         inputTexts.append(tempInput)
#         i += 1
#     print("done lamma2 preds")

#     filePath = "results/" + model_name + "_predictions.json"
#     # Save results
#     with open(filePath, "w") as f:
#         json.dump(predictions_smollm, f, indent=4)
    
#     inputPath = "results/" + model_name + "_input.json"
#     # Save results
#     with open(inputPath, "w") as f:
#         json.dump(inputTexts, f, indent=4)

#     print("Inference complete. Predictions saved.")

def generate_summaries(tokenizer, dev_data, model_name, summary_method):
    num_samples = 20
    print(f"Predicting with {model_name}...\n")

    predictions = []
    inputs = []

    for i, ex in enumerate(dev_data[:num_samples]):
        start_time = time.time()

        # 1) Generate the raw output (which will be something like ["Your summary"]).
        if summary_method == "first_500":
            raw_pred, inp = generate_summary_first_500(model_name, ex["article"])
        elif summary_method == "chunks_100":
            raw_pred, inp = generate_summary_chunks_100(model_name, ex["article"])

        elapsed = time.time() - start_time

        # 2) Parse the LLM’s JSON array to get a clean string.
        try:
            # raw_pred should look like '["Some text"]'
            # Attempt to parse as JSON
            summary_list = json.loads(raw_pred.strip())
            # We expect a single string at index [0].
            parsed_pred = summary_list[0]
        except (json.JSONDecodeError, IndexError):
            # If parsing fails, just use the raw output
            parsed_pred = raw_pred.strip()

        # 3) Append the cleaned summary + input text
        predictions.append(parsed_pred)
        inputs.append(inp)

        # 4) Log to W&B, measuring length based on the cleaned summary
        wandb.log({
            "sample_index": i,
            "input_length": len(inp.split()),
            "summary_length": len(parsed_pred.split()),
            "inference_time_sec": elapsed
        })

        # Print sample info
        print(f"\n--- Sample {i+1} ---")
        print("Input (first 3 lines):")
        print("\n".join(inp.splitlines()[:3])[:300])
        print("\nModel Output:")
        print(parsed_pred[:300] + ("..." if len(parsed_pred) > 300 else ""))
        print("----------------------------")

    # Save the final data
    BASE_DIR = Path(__file__).resolve().parents[1]
    RESULTS_DIR = BASE_DIR / "results"
    RESULTS_DIR.mkdir(exist_ok=True)

    predictions_file = RESULTS_DIR / "test_predictions.json"
    inputs_file = RESULTS_DIR / "test_input.json"

    with open(predictions_file, "w") as f:
        print(f"Saving predictions to {f.name}")
        json.dump(predictions, f, indent=4)

    with open(inputs_file, "w") as f:
        print(f"Saving inputs to {f.name}")
        json.dump(inputs, f, indent=4)

    print("Inference complete. Predictions saved.")