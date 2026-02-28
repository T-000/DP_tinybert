from transformers import AutoTokenizer, AutoModelForSequenceClassification

def load_tinybert(model_name: str, num_labels: int):
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    return tok, model