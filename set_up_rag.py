import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm 

print("Setting up RAG index...")

# --- Config ---
JSONL_FILE = "harshit_abighya_rag_chat_10k_hinglish (1).jsonl"
INDEX_FILE = "rag_10k.index"
META_FILE  = "rag_10k_meta.jsonl"
MODEL_NAME = "all-MiniLM-L6-v2"

def build_rag_index(jsonl_path=JSONL_FILE, index_path=INDEX_FILE, meta_path=META_FILE):
    embedder = SentenceTransformer(MODEL_NAME)

    # Load instructions
    instructions = []
    meta_data = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            if "instruction" in obj:
                instructions.append(obj["instruction"])
                meta_data.append(obj)

    print(f"Loaded {len(instructions):,} instructions")

    # Embed
    print("Generating embeddings...")
    vectors = embedder.encode(instructions, show_progress_bar=True, batch_size=64)
    vectors = np.array(vectors).astype("float32")

    # FAISS index
    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors)

    # Save
    print("Saving index and metadata...")
    faiss.write_index(index, index_path)

    with open(meta_path, "w", encoding="utf-8") as f:
        for record in meta_data:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Saved index to: {index_path}")
    print(f"Saved metadata to: {meta_path}")

    return index, embedder, meta_data

if __name__ == "__main__":
    build_rag_index()
    print("RAG index built successfully!")