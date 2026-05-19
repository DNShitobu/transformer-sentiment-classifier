"""
Hugging Face Space — Sentiment Classifier
==========================================
Gradio demo for the fine-tuned DistilBERT sentiment classifier.
Deploy this as a Hugging Face Space (SDK: Gradio).
"""

import gradio as gr
from transformers import pipeline

# Replace with your actual HF model repo once pushed
MODEL_ID = "distilbert-base-uncased-finetuned-sst-2-english"  # placeholder until your model is pushed

print(f"Loading model: {MODEL_ID}")
pipe = pipeline("text-classification", model=MODEL_ID)

EXAMPLES = [
    "This movie was absolutely fantastic! Best film of the year.",
    "I hated every minute of it. Boring, predictable, and poorly acted.",
    "An okay film. Nothing special but watchable.",
    "Phenomenal storytelling and brilliant performances throughout.",
    "Complete waste of time and money. Avoid at all costs.",
    "The cinematography was stunning but the plot fell flat.",
]

def classify(text):
    if not text.strip():
        return "Please enter some text.", ""
    result = pipe(text)[0]
    label  = result["label"].upper()
    score  = result["score"]
    emoji  = "😊 POSITIVE" if label == "POSITIVE" else "😞 NEGATIVE"
    bar    = "█" * int(score * 20) + "░" * (20 - int(score * 20))
    detail = f"{bar}  {score:.1%} confidence"
    return emoji, detail

with gr.Blocks(title="Sentiment Classifier", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🎭 Sentiment Classifier
    **Fine-tuned DistilBERT** · Binary sentiment analysis (positive / negative)

    Type any text and the model will predict its sentiment with confidence score.
    """)

    with gr.Row():
        with gr.Column(scale=2):
            text_box = gr.Textbox(
                label="Enter text to classify",
                placeholder="e.g. This was the best experience of my life!",
                lines=4,
            )
            submit_btn = gr.Button("Classify Sentiment", variant="primary")

        with gr.Column(scale=1):
            label_out  = gr.Textbox(label="Prediction", interactive=False)
            conf_out   = gr.Textbox(label="Confidence", interactive=False)

    gr.Examples(
        examples=[[e] for e in EXAMPLES],
        inputs=text_box,
        label="Try these examples",
    )

    submit_btn.click(fn=classify, inputs=text_box, outputs=[label_out, conf_out])
    text_box.submit(fn=classify, inputs=text_box, outputs=[label_out, conf_out])

    gr.Markdown("""
    ---
    **Model:** DistilBERT fine-tuned on SST-2 · **Built by** [Dnshitobu](https://github.com/Dnshitobu)
    """)

demo.launch()
