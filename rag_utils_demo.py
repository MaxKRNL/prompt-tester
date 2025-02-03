# rag_utils_demo.py

import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Tuple

INDEX_FILE = "faiss_index.bin"
DOCS_FILE = "doc_texts.pkl"

faiss_index = None
doc_texts = []
embedder = None

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def load_precomputed_index():
    """
    Loads the precomputed FAISS index & doc chunks from disk,
    and initializes the embedder for query encoding.
    """
    global faiss_index, doc_texts, embedder

    print("Loading precomputed FAISS index & doc texts...")
    faiss_index = faiss.read_index(INDEX_FILE)
    with open(DOCS_FILE, "rb") as f:
        doc_texts = pickle.load(f)

    embedder = SentenceTransformer(EMBED_MODEL_NAME)
    print(f"Loaded {len(doc_texts)} doc chunks. FAISS index is ready.")

def retrieve_context(query: str, top_k: int = 3) -> List[Tuple[str, float]]:
    """
    Encodes the user query, searches the index, returns list of (chunk, score).
    """
    global faiss_index, doc_texts, embedder
    if faiss_index is None or embedder is None:
        raise ValueError("FAISS index not loaded. Call load_precomputed_index() first.")

    query_emb = embedder.encode([query]).astype("float32")
    scores, indices = faiss_index.search(query_emb, top_k)

    results = []
    for i, idx in enumerate(indices[0]):
        chunk_text = doc_texts[idx]
        sim_score = float(scores[0][i])
        results.append((chunk_text, sim_score))

    return results
