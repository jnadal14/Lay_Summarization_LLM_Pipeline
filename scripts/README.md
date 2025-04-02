# Scripts Directory

## **Overview**
This folder contains all the scripts required to run the full pipeline for the BioLaySumm 2025 shared task (Milestone 4). The scripts are modular, allowing each component (preprocessing, inference, and evaluation) to be run independently or as part of the main.py pipeline.

In this milestone, we no longer fine-tune Hugging Face models — instead, we use a local LLaMA 2 model via Ollama for zero-shot inference.
---

## **Directory Structure**
```plaintext
COLX_531_Project_Cole-Daoming-Jacob-Juan/milestone4/scripts/
├── preprocess.py         # Loads & preprocesses datasets from Hugging Face (returns train/dev sets in-memory)
├── inference.py          # Generates summaries using selected LLM with truncated input prompts
├── evaluate_model.py     # Evaluates predictions using ROUGE and BLEU (BERTScore removed for memory efficiency)
├── train.py              # Leftover from baseline; unused in this milestone but retained for compatibility
├── utils.py              # Shared helper functions (e.g., saving outputs, W&B logging, etc.)