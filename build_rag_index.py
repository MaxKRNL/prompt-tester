# build_rag_index.py

import os
import re
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

CHUNK_SIZE = 500
OVERLAP = 50
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

INDEX_FILE = "faiss_index.bin"
DOCS_FILE = "doc_texts.pkl"

def chunk_text(text: str, chunk_size: int, overlap: int):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += (chunk_size - overlap)
    return chunks

def load_company_data(folder: str = "company_data"):
    texts = []
    for fn in os.listdir(folder):
        if fn.endswith(".txt"):
            path = os.path.join(folder, fn)
            with open(path, "r", encoding="utf-8") as f:
                texts.append(f.read())
    return texts

def main():
    # 1. Load data
    raw_texts = load_company_data("company_data")
    print(f"Found {len(raw_texts)} text file(s) in 'company_data/'.")

    # 2. Chunk data
    all_chunks = []
    for txt in raw_texts:
        # optionally remove large whitespace
        txt = re.sub(r"\s+", " ", txt).strip()
        chunks = chunk_text(txt, CHUNK_SIZE, OVERLAP)
        all_chunks.extend(chunks)

    print(f"Created {len(all_chunks)} total chunks.")

    # 3. Embed chunks
    print("Embedding chunks with SentenceTransformer...")
    embedder = SentenceTransformer(EMBED_MODEL_NAME)
    embeddings = embedder.encode(all_chunks, show_progress_bar=True)
    embeddings = np.array(embeddings, dtype="float32")

    # 4. Build FAISS index
    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(embeddings)
    print(f"FAISS index size: {index.ntotal}")

    # 5. Save index + document texts
    faiss.write_index(index, INDEX_FILE)
    with open(DOCS_FILE, "wb") as f:
        pickle.dump(all_chunks, f)

    print(f"Saved FAISS index to '{INDEX_FILE}', doc chunks to '{DOCS_FILE}'.")

if __name__ == "__main__":
    main()
