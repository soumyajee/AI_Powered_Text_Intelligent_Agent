import os
import faiss
import numpy as np
import json
from threading import Lock
from typing import List
from app.config import VECTOR_DIM
from app.service.embeddings import embed

# -----------------------------
# Paths
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "..", "data")
INDEX_PATH = os.path.join(DATA_DIR, "faiss.index")
DOCS_PATH = os.path.join(DATA_DIR, "documents.json")

os.makedirs(DATA_DIR, exist_ok=True)

# -----------------------------
# Thread safety
# -----------------------------
_lock = Lock()

# -----------------------------
# Helper functions
# -----------------------------
def _load_index() -> faiss.IndexFlatL2:
    if os.path.exists(INDEX_PATH):
        return faiss.read_index(INDEX_PATH)
    else:
        return faiss.IndexFlatL2(VECTOR_DIM)

def _save_index(index: faiss.IndexFlatL2) -> None:
    faiss.write_index(index, INDEX_PATH)

def _load_documents() -> List[str]:
    if os.path.exists(DOCS_PATH):
        with open(DOCS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def _save_documents(docs: List[str]) -> None:
    with open(DOCS_PATH, "w", encoding="utf-8") as f:
        json.dump(docs, f)

# -----------------------------
# Add document
# -----------------------------
def add_document(text: str) -> None:
    vector = embed(text)
    if len(vector) != VECTOR_DIM:
        raise ValueError(f"Embedding dimension mismatch: expected {VECTOR_DIM}, got {len(vector)}")

    vector_np = np.array([vector], dtype="float32")

    with _lock:
        # Load current state
        index = _load_index()
        documents = _load_documents()

        # Add new vector and document
        index.add(vector_np)
        documents.append(text)

        # Save back
        _save_index(index)
        _save_documents(documents)

# -----------------------------
# Semantic search
# -----------------------------
def search_similar(query: str, top_k: int = 5) -> List[str]:
    vector = embed(query)
    if len(vector) != VECTOR_DIM:
        raise ValueError(f"Embedding dimension mismatch: expected {VECTOR_DIM}, got {len(vector)}")

    vector_np = np.array([vector], dtype="float32")

    with _lock:
        index = _load_index()
        if index.ntotal == 0:
            return []

        documents = _load_documents()
        k = min(top_k, index.ntotal)
        distances, indices = index.search(vector_np, k)

        results = []
        for i in indices[0]:
            if i != -1 and i < len(documents):  # -1 can appear if k > ntotal
                results.append(documents[i])

        return results
