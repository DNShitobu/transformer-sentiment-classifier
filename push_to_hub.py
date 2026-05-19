"""
push_to_hub.py
==============
Uploads your fine-tuned DistilBERT model to the Hugging Face Hub,
then it will appear at: https://huggingface.co/Dnshitobu/distilbert-sentiment

Usage:
    1. Run train.py to fine-tune the model (saves to ./sentiment-model/)
    2. huggingface-cli login     # paste your HF write token
    3. python push_to_hub.py
"""

from transformers import AutoTokenizer, AutoModelForSequenceClassification

HF_USERNAME  = "Dnshitobu"
MODEL_REPO   = f"{HF_USERNAME}/distilbert-sentiment"
MODEL_DIR    = "./sentiment-model"

print(f"Pushing model to: https://huggingface.co/{MODEL_REPO}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model     = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)

tokenizer.push_to_hub(MODEL_REPO)
model.push_to_hub(MODEL_REPO)

print(f"\nDone! Your model is live at:")
print(f"  https://huggingface.co/{MODEL_REPO}")
print(f"\nTo use it:")
print(f"  from transformers import pipeline")
print(f"  pipe = pipeline('text-classification', model='{MODEL_REPO}')")
print(f"  print(pipe('This movie was amazing!'))")
print(f"\nUpdate MODEL_ID in app.py in your HF Space to '{MODEL_REPO}'")
print(f"  Space: https://huggingface.co/spaces/Dnshitobu/sentiment-classifier")
