import torch
from torch import nn
import numpy as np
from transformers import AutoTokenizer,AutoModelForSequenceClassification
from config import Settings


# Initialize settings
settings = Settings()

softmax = nn.Softmax(dim=1)

# Load configuration from settings
model_name = settings.model_name
num_labels = settings.num_labels
hugging_face_api_key = settings.hugging_face_api_key
categories = settings.categories


def get_device():
    """Get the available device (GPU if available, else CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model_and_tokenizer(model_name, num_labels):
    """Load the pre-trained model and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name,num_labels=num_labels)
    model.to(get_device())
    return model, tokenizer



def preprocess_texts(texts, tokenizer, max_length=256):
    """Tokenize and preprocess texts for text input."""
    encodings = tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
        add_special_tokens=True,
    )
    return encodings

def make_prediction(model, encodings):
    """Make predictions using the model and preprocessed encodings."""
    model.eval()
    with torch.no_grad():
        input_ids = encodings['input_ids'].to(get_device())
        attention_mask = encodings['attention_mask'].to(get_device())
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probabilities = softmax(logits)
        predicted_classes, probabilities = torch.max(probabilities, dim=1)
    return predicted_classes.cpu().numpy(), probabilities.cpu().numpy()


def predict(texts: str):
    model, tokenizer = load_model_and_tokenizer(model_name, num_labels)
    encodings = preprocess_texts(texts, tokenizer=tokenizer)
    probability, predicted_class = make_prediction(model, encodings)

    return round(probability.item(),4), categories[predicted_class.item()]


if __name__ == "__main__":
    sample_texts = "BurnaBoy is realy an incredible singer. You know from the quality of work he has put out over the years"
    print(predict(texts=sample_texts))