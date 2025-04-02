# Milestone 5: Refined LLaMA 2 Inference & Lay Summarization

## **Overview**

In Milestone 5, we improve the quality and usability of our lay summarization pipeline by refining the inference process using **LLaMA 2 via Ollama**. Building on Milestone 4, our focus shifts to **prompt engineering**, **clean output formatting**, and **controlled evaluation**, while preserving full compatibility with Weights & Biases (W&B) and Hugging Face evaluation metrics.

We split compute responsibilities: inference runs efficiently on Jacob’s Apple Silicon MacBook using Ollama, while evaluation is performed on Cole’s higher-performance GPU machine due to BERTScore memory constraints.

---

## **Key Improvements Over Milestone 4**

- **Improved Prompt Engineering**  
  Prompts now instruct the model to produce concise, plain-language summaries with minimal biomedical jargon, formatted as a clean JSON list. This reduces verbosity and ensures consistency for downstream evaluation.

- **Streaming Inference Output**  
  Input excerpts and model summaries are printed in real-time, enabling easier inspection and debugging during generation.

- **Path Robustness with `pathlib`**  
  All file paths are relative to the project root, ensuring smooth cross-platform operation regardless of environment.

- **Evaluation Toggle**  
  A new `shouldEvaluate` flag allows evaluation to be run independently, enabling division of work between team members.

- **Weights & Biases Integration**  
  All inference and evaluation results are logged to W&B with dynamically named runs and optional per-sample metrics.

---

### **Directory Structure**
```plaintext
COLX_531_Project_Cole-Daoming-Jacob-Juan/milestone5/
├── main.py                      # Main pipeline script
├── scripts/
│   ├── inference.py             # Inference using LLaMA 2 + Ollama
│   ├── evaluate_model.py        # Evaluation (ROUGE, BLEU, BERTScore)
│   └── preprocess.py            # Preprocesses and splits dataset
├── results/
│   ├── test_predictions.json    # JSON summaries from LLaMA 2
│   ├── test_input.json          # Corresponding input abstracts
├── wandb/                       # Weights & Biases logs
└── README.md                    # This file
```