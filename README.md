# Transformer Sentiment Classifier

Fine-tuning DistilBERT on SST-2 for binary sentiment analysis, with a live Gradio demo on Hugging Face Spaces.

## Overview

| Detail | Value |
|--------|-------|
| Base model | distilbert-base-uncased |
| Dataset | SST-2 (67K training sentences) |
| Task | Binary classification (positive / negative) |
| Framework | HuggingFace Transformers + PyTorch |
| Accuracy | ~92-93% on SST-2 validation |

## Project Structure

```
transformer-sentiment-classifier/
├── train.py          # Fine-tune DistilBERT on SST-2
├── push_to_hub.py    # Push trained model to HF Hub
├── app.py            # Gradio Space demo
├── requirements.txt
└── README.md
```

## Getting Started

```bash
pip install -r requirements.txt

# Train the model (~30 min on GPU, ~2 hrs CPU)
python train.py

# Log in to Hugging Face
huggingface-cli login

# Push to Hub
python push_to_hub.py
```

## Key Concepts

- Transfer learning: adapting a pre-trained language model to a new task
- Tokenisation with WordPiece (DistilBERT)
- Fine-tuning with HuggingFace Trainer API
- Evaluation: accuracy, F1, classification report
- Deploying models to the HF Hub and Spaces
