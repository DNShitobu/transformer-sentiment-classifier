"""
Transformer Sentiment Classifier
==================================
Fine-tunes DistilBERT on the SST-2 dataset for binary sentiment classification.

Requirements:
    pip install transformers datasets torch accelerate scikit-learn

Usage:
    python train.py

After training, run: python push_to_hub.py
"""

import os
import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
)
from sklearn.metrics import accuracy_score, f1_score, classification_report

MODEL_NAME    = "distilbert-base-uncased"
OUTPUT_DIR    = "./sentiment-model"
NUM_EPOCHS    = 3
BATCH_SIZE    = 16
LEARNING_RATE = 2e-5
MAX_LENGTH    = 128
SEED          = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

print("=" * 60)
print("TRANSFORMER SENTIMENT CLASSIFIER")
print(f"Model  : {MODEL_NAME}")
print(f"Task   : Binary sentiment (SST-2)")
print(f"Device : {'GPU' if torch.cuda.is_available() else 'CPU'}")
print("=" * 60)

# 1. Load dataset
print("\nLoading SST-2 dataset...")
dataset = load_dataset("glue", "sst2")
print(f"Train: {len(dataset['train']):,}  Validation: {len(dataset['validation']):,}")

# 2. Tokenise
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize(batch):
    return tokenizer(batch["sentence"], truncation=True, max_length=MAX_LENGTH)

tokenized = dataset.map(tokenize, batched=True, remove_columns=["sentence","idx"])
tokenized = tokenized.rename_column("label", "labels")
tokenized.set_format("torch")
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# 3. Load model
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=2,
    id2label={0:"negative", 1:"positive"},
    label2id={"negative":0, "positive":1},
)

# 4. Metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(labels, preds), "f1": f1_score(labels, preds)}

# 5. Train
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE * 2,
    learning_rate=LEARNING_RATE,
    weight_decay=0.01,
    warmup_ratio=0.1,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    logging_steps=100,
    fp16=torch.cuda.is_available(),
    seed=SEED,
    report_to="none",
)

trainer = Trainer(
    model=model, args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
)

print(f"\nTraining {NUM_EPOCHS} epochs (lr={LEARNING_RATE}, batch={BATCH_SIZE})...")
trainer.train()

# 6. Evaluate
results = trainer.evaluate()
print(f"\nValidation Accuracy : {results['eval_accuracy']:.4f}")
print(f"Validation F1       : {results['eval_f1']:.4f}")

preds_output = trainer.predict(tokenized["validation"])
preds  = np.argmax(preds_output.predictions, axis=-1)
labels = preds_output.label_ids
print("\n" + classification_report(labels, preds, target_names=["negative","positive"]))

# 7. Save
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"\nModel saved to {OUTPUT_DIR}/")

# 8. Quick inference
from transformers import pipeline
pipe = pipeline("text-classification", model=OUTPUT_DIR, tokenizer=OUTPUT_DIR)
tests = [
    "This movie was absolutely fantastic!",
    "I hated every single minute of this film.",
    "One of the best performances I have ever seen.",
    "Terrible acting, boring plot, waste of time.",
]
print("\nQuick inference test:")
for s in tests:
    r = pipe(s)[0]
    emoji = "😊" if r["label"]=="positive" else "😞"
    print(f"  {emoji} [{r['label']} {r['score']:.2%}] {s}")

print("\nRun 'python push_to_hub.py' to publish to Hugging Face.")
