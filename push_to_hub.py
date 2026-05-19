"""
push_to_hub.py
==============
Uploads the fine-tuned model and tokenizer to Hugging Face Hub,
then creates a Gradio Space demo.

Usage:
    huggingface-cli login     # enter your HF write token
    python push_to_hub.py
"""

from transformers import AutoTokenizer, AutoModelForSequenceClassification

HF_USERNAME  = "YOUR_HF_USERNAME"   # <-- replace with your HF username
MODEL_REPO   = f"{HF_USERNAME}/distilbert-sentiment"
MODEL_DIR    = "./sentiment-model"

print(f"Pushing model to: {MODEL_REPO}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model     = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)

tokenizer.push_to_hub(MODEL_REPO)
model.push_to_hub(MODEL_REPO)

print(f"Done! Model live at: https://huggingface.co/{MODEL_REPO}")
print(f"Pipeline: from transformers import pipeline")
print(f"          pipe = pipeline('text-classification', model='{MODEL_REPO}')")
