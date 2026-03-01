from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_NAME = "prajjwal1/bert-tiny"

def load_tinybert(num_labels: int):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_labels
    )
    return tokenizer, model