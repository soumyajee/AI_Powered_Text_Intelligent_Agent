import os
import faiss
import numpy as np
import json
from threading import Lock
from typing import List, Dict, Any
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
def _load_index() -> faiss.Index:
    """
    Use cosine similarity by:
      - IndexFlatIP (inner product)
      - L2-normalize vectors before add/search
    With normalization, returned scores are cosine similarity in [-1, 1]
    and will NOT exceed 1.
    """
    if os.path.exists(INDEX_PATH):
        idx = faiss.read_index(INDEX_PATH)

        # Safety: if an old L2 index exists, it will give distances, not cosine.
        # You should rebuild once if you changed index type.
        if not isinstance(idx, faiss.IndexFlatIP):
            raise RuntimeError(
                "Existing index is not IndexFlatIP. Delete data/faiss.index and rebuild."
            )
        return idx

    return faiss.IndexFlatIP(VECTOR_DIM)

def _save_index(index: faiss.Index) -> None:
    faiss.write_index(index, INDEX_PATH)

def _load_documents() -> List[str]:
    if os.path.exists(DOCS_PATH):
        with open(DOCS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def _save_documents(docs: List[str]) -> None:
    with open(DOCS_PATH, "w", encoding="utf-8") as f:
        json.dump(docs, f, ensure_ascii=False)

def _embed_np(text: str) -> np.ndarray:
    vec = embed(text)
    if len(vec) != VECTOR_DIM:
        raise ValueError(f"Embedding dimension mismatch: expected {VECTOR_DIM}, got {len(vec)}")
    arr = np.array([vec], dtype="float32")  # shape (1, dim)
    # IMPORTANT: normalize so inner product == cosine similarity
    faiss.normalize_L2(arr)
    return arr

# -----------------------------
# Add document
# -----------------------------
def add_document(text: str) -> None:
    vector_np = _embed_np(text)

    with _lock:
        index = _load_index()
        documents = _load_documents()

        index.add(vector_np)
        documents.append(text)

        _save_index(index)
        _save_documents(documents)

# -----------------------------
# Semantic search (returns cosine scores, never > 1)
# -----------------------------
def search_similar(query: str, top_k: int = 5, min_score: float = 0.75) -> List[Dict[str, Any]]:
    """
    Returns list of {"text": ..., "score": ...}
    score is cosine similarity in [-1, 1] and will not exceed 1.
    """
    query_np = _embed_np(query)

    with _lock:
        index = _load_index()
        if index.ntotal == 0:
            return []

        documents = _load_documents()
        k = min(top_k, index.ntotal)

        scores, indices = index.search(query_np, k)  # cosine similarities

        results: List[Dict[str, Any]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1 or idx >= len(documents):
                continue

            score_f = float(score)
            if score_f < min_score:
                continue

            results.append({"text": documents[idx], "score": score_f})

        return results

# -----------------------------
# Optional: one-time rebuild helper (run once after deleting old L2 index)
# -----------------------------
def rebuild_index_from_docs() -> None:
    """
    If you previously used IndexFlatL2, do this once:
      1) delete data/faiss.index
      2) call rebuild_index_from_docs()
    """
    docs = _load_documents()
    index = faiss.IndexFlatIP(VECTOR_DIM)

    for d in docs:
        v = _embed_np(d)  # normalized
        index.add(v)

    _save_index(index)
