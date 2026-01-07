from fastapi import FastAPI
from app.schemas import TextRequest, SearchRequest, AnalyzeResponse, SummaryResponse
from app.service.sentiment import get_sentiment
from app.service.keywords import extract_keywords
from app.service.summarizer import summarize
from app.vectorstore.faiss_store import add_document, search_similar

app = FastAPI(
    title="AI-Powered Text Intelligence API",
    version="1.0.0"
)

@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: TextRequest):
    return {
        "sentiment": get_sentiment(req.text),
        "keywords": extract_keywords(req.text)
    }

@app.post("/summarize", response_model=SummaryResponse)
def summarize_text(req: TextRequest):
    return {"summary": summarize(req.text)}

@app.post("/add-document")
def add_doc(req: TextRequest):
    add_document(req.text)
    return {"status": "document added successfully"}

@app.post("/semantic-search")
def semantic_search(req: SearchRequest):
    matches = search_similar(req.query, req.top_k or 5)
    return {"matches": matches}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)