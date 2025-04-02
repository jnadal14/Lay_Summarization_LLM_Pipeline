# COLX 531/585 Group Project - Lay Summarization of Biomedical Research Articles and Radiology

## Group Repository: Daoming Liu, Jacob Nadal, Cole Piche, Juan Tampubolon

### **Project Overview**
This repository contains our group project for COLX 531 – Neural Machine Translation, focusing on lay summarization of biomedical research papers and radiology findings. Our goal is to generate concise, plain-language summaries from scientific or clinical abstracts so that non-expert audiences can easily understand key insights. We use LLaMA 2 via Ollama, along with streaming output and improved prompt engineering, to produce short, jargon-minimized summaries in JSON format for clear downstream evaluation.

## **Key Scripts**

`main.py` is the central script that orchestrates the entire pipeline:

- **Preprocessing**  
  It first calls `scripts/preprocess.py` (if needed) to load and clean the dataset, remove irrelevant fields, and split the data into train/test.

- **Inference**  
  Next, it invokes `scripts/inference.py` to run the LLaMA 2 model. We pass curated prompts that ensure the summaries are short, user-friendly, and consistently formatted.

- **Evaluation**  
 If a flag (e.g., shouldEvaluate) is set to True, `scripts/evaluate_model.py` is called to compute metrics such as ROUGE, BLEU, and BERTScore. The results can be logged locally or to Weights & Biases for collaborative tracking.

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
├── REQUIREMENTS.txt             # List of packages required to run main.py
└── README.md                    # This file
```
