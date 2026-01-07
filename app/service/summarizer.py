from transformers import pipeline
from app.config import MODEL_SUMMARY

_summarizer = pipeline("summarization", model=MODEL_SUMMARY)

def summarize(text: str) -> str:
    summary = _summarizer(
        text,
        max_length=120,
        min_length=30,
        do_sample=False
    )
    return summary[0]["summary_text"]
