import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Config
INDEX_PATH = "rag_10k.index"
META_PATH = "rag_10k_meta.jsonl"
MODEL_NAME = "all-MiniLM-L6-v2"

# Load model and index once
embedder = SentenceTransformer(MODEL_NAME)
index = faiss.read_index(INDEX_PATH)
with open(META_PATH, "r", encoding="utf-8") as f:
    meta = [json.loads(line) for line in f]

def retrieve_memories(query, k=5):
    """Returns top-k relevant instruction-response dicts from memory."""
    query_vec = embedder.encode([query], normalize_embeddings=True)
    _, idxs = index.search(np.array(query_vec, dtype="float32"), k)
    return [meta[i] for i in idxs[0]]
