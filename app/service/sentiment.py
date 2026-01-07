from transformers import pipeline
from app.config import MODEL_SENTIMENT

_sentiment_model = pipeline("sentiment-analysis", model=MODEL_SENTIMENT)

def get_sentiment(text: str) -> str:
    result = _sentiment_model(text)[0]
    return result["label"].lower()
