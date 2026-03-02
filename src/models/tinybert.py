from transformers import BertTokenizer, BertForSequenceClassification

MODEL_NAME = "prajjwal1/bert-tiny"

def load_tinybert(num_labels: int):
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    model = BertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_labels,
    )
    return tokenizer, model