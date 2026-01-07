import spacy

nlp = spacy.load("en_core_web_sm")

def extract_keywords(text: str, top_k: int = 5) -> list[str]:
    doc = nlp(text)
    keywords = [
        token.text.lower()
        for token in doc
        if token.pos_ in {"NOUN", "PROPN", "ADJ"}
        and not token.is_stop
    ]
    return list(dict.fromkeys(keywords))[:top_k]
