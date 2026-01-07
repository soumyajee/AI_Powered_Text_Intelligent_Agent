# AI-Powered Text Intelligence API

An end-to-end NLP-based API service built with **FastAPI**, **FAISS**, and **embedding models**, providing:

- Sentiment analysis
- Keyword extraction
- Text summarization
- Semantic search with persistent FAISS index

---

## 🚀 Features

1. **Text Sentiment & Keyword Analysis**
   - Detects sentiment (`positive`, `negative`, `neutral`) of input text.
   - Extracts top keywords based on parts-of-speech using SpaCy.

2. **Text Summarization**
   - Uses pre-trained Transformer models (T5, BART, GPT) to summarize text.

3. **Semantic Search**
   - Stores embeddings in FAISS index.
   - Supports similarity search with top-K results.
   - Persists FAISS index and document mapping across server restarts.

4. **Deployment**
   - Containerized using Docker.
   - Swagger UI documentation available at `/docs`.

---

## 📁 Project Structure

AI_Powered_Text_Intelligence_API/
├── app/
│ ├── main.py # FastAPI entry point
│ ├── schemas.py # Pydantic request/response models
│ ├── service/
│ │ ├── embeddings.py # Embedding model
│ │ ├── sentiment.py # Sentiment analysis
│ │ ├── keywords.py # Keyword extraction
│ │ └── summarizer.py # Text summarization
│ └── vectorstore/
│ └── faiss_store.py # FAISS index management
├── data/
│ ├── faiss.index # Persistent FAISS index
│ └── documents.json # Persistent document mapping
├── Dockerfile
├── requirements.txt
└── README.md

---

## ⚙️ Setup Instructions

### 1. Clone the repository

```bash
git clone <repo-url>
cd AI_Powered_Text_Intelligence_API
2. Create virtual environment & install dependencies
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows

pip install -r requirements.txt

3. Run FastAPI server
uvicorn app.main:app --reload
📝 API Endpoints
1. Analyze Text

POST /analyze

Request
{
  "text": "I love working with AI! It makes everything efficient."
}

Response
{
  "sentiment": "positive",
  "keywords": ["AI", "efficient", "love"]
}

2. Summarize Text

POST /summarize

Request
{
  "text": "Artificial intelligence is transforming the way businesses operate..."
}

Response
{
  "summary": "AI is changing business operations significantly."
}

3. Add Document (for Semantic Search)

POST /documents
{
  "text": "AI improves productivity and efficiency at work."
}
Response
{
  "message": "Document added successfully."
}
4. Semantic Search

POST /semantic-search

Request
{
  "query": "AI helps people work efficiently",
  "top_k": 3
}
Response
{
  "matches": [
    "AI improves productivity and efficiency at work",
    "Machine learning helps automate tasks"
  ]
}
🔧 FAISS Semantic Search Fix

Problem: FAISS only stores vectors, not document text. Previous versions returned empty results after restart.

Solution:

Persist FAISS index to disk (faiss.index)

Persist corresponding document mapping (documents.json)

Load both at startup

This ensures semantic search works reliably across server restarts.
🐳 Docker Deployment

Dockerfile example:
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
Build and run:
docker build -t text-intelligence-api .
docker run -p 8000:8000 text-intelligence-api
