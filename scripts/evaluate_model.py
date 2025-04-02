import json
import evaluate
from pathlib import Path
from preprocess import load_and_preprocess

# Load official evaluation metrics
rouge = evaluate.load("rouge")
bleu = evaluate.load("bleu")
bertscore = evaluate.load("bertscore")

# Evaluate models
def evaluate_predictions(predictions, references, model_name):
    rouge_scores = rouge.compute(predictions=predictions, references=references)
    bleu_score = bleu.compute(predictions=predictions, references=references)
    bert_score = bertscore.compute(predictions=predictions, references=references, lang="en")

    print(f"\n### {model_name} Performance ###")
    print(f"ROUGE-1: {rouge_scores['rouge1']:.4f}")
    print(f"ROUGE-2: {rouge_scores['rouge2']:.4f}")
    print(f"ROUGE-L: {rouge_scores['rougeL']:.4f}")
    print(f"BLEU: {bleu_score['bleu']:.4f}")
    print(f"BERTScore (F1): {sum(bert_score['f1']) / len(bert_score['f1']):.4f}")

    return {
        "ROUGE-1": rouge_scores["rouge1"],
        "ROUGE-2": rouge_scores["rouge2"],
        "ROUGE-L": rouge_scores["rougeL"],
        "BLEU": bleu_score["bleu"],
        "BERTScore (F1)": sum(bert_score["f1"]) / len(bert_score["f1"])
    }

def evaluate_model(model_name):
    # Resolve base path relative to milestone5/
    BASE_DIR = Path(__file__).resolve().parents[1]
    results_path = BASE_DIR / "results" / f"test_predictions.json"

    with open(results_path, "r") as f:
        predictions = json.load(f)

    _, dev_data = load_and_preprocess()
    references = [ex["summary"] for ex in dev_data[:len(predictions)]]

    return evaluate_predictions(predictions, references, model_name)