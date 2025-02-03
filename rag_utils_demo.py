# rag_utils_demo.py

import os
import faiss
import numpy as np
import re
from typing import List, Tuple
from sentence_transformers import SentenceTransformer

faiss_index = None
doc_texts = []
embedder = None

CHUNK_SIZE = 500  # Characters per chunk
OVERLAP = 50      # Overlap between chunks
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += (chunk_size - overlap)
    return chunks

def load_company_data(data_folder: str = "company_data") -> List[str]:
    texts = []
    for fn in os.listdir(data_folder):
        if fn.endswith(".txt"):
            file_path = os.path.join(data_folder, fn)
            with open(file_path, "r", encoding="utf-8") as f:
                texts.append(f.read())
    return texts

def initialize_faiss_demo(data_folder: str = "company_data"):
    """
    Build FAISS index from .txt in company_data/ for RAG.
    """
    global faiss_index, doc_texts, embedder

    raw_texts = load_company_data(data_folder)
    all_chunks = []
    for txt in raw_texts:
        # optionally remove newlines, etc.
        txt = re.sub(r"\s+", " ", txt).strip()
        chunks = chunk_text(txt, CHUNK_SIZE, OVERLAP)
        all_chunks.extend(chunks)

    doc_texts = all_chunks

    embedder = SentenceTransformer(EMBED_MODEL_NAME)
    embeddings = embedder.encode(doc_texts, show_progress_bar=False)
    embeddings = np.array(embeddings, dtype="float32")

    d = embeddings.shape[1]
    faiss_index = faiss.IndexFlatIP(d)
    faiss_index.add(embeddings)

def retrieve_context(query: str, top_k: int = 3) -> List[Tuple[str, float]]:
    """
    Embed query, search FAISS, return (chunk, score).
    """
    global faiss_index, doc_texts, embedder
    if faiss_index is None or embedder is None:
        raise ValueError("FAISS index not initialized. Call initialize_faiss_demo() first.")

    query_emb = embedder.encode([query])
    query_emb = np.array(query_emb, dtype="float32")

    scores, indices = faiss_index.search(query_emb, top_k)
    results = []
    for i, idx in enumerate(indices[0]):
        results.append((doc_texts[idx], float(scores[0][i])))

    return results
