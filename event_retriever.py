import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

EVENT_FILE = "event_memories.json"
MODEL_NAME = "all-MiniLM-L6-v2"
THRESHOLD = 0.65  # adjust as needed

# Load data and build index once
with open(EVENT_FILE, "r", encoding="utf-8") as f: 
    event_data = json.load(f)
    event_keys = [event["title"] for event in event_data]


print(f"Loaded {len(event_data)} events from {EVENT_FILE}")



# Initialize model
embedder = SentenceTransformer(MODEL_NAME)
event_embeddings = embedder.encode(event_keys, normalize_embeddings=True)
event_embeddings = np.array(event_embeddings).astype("float32")

# Build FAISS index
index = faiss.IndexFlatIP(event_embeddings.shape[1])
index.add(event_embeddings)


def retrieve_event_memory(query, top_k=1):
    query_embedding = embedder.encode([query], normalize_embeddings=True).astype("float32")
    scores, idxs = index.search(query_embedding, top_k)

    if scores[0][0] >= THRESHOLD:
        matched_event = event_data[idxs[0][0]]
        result = {
            "event": matched_event["title"],
            "date": matched_event["date"],
            "description": matched_event["description"] 

        } 
        print(f"Matched event: {matched_event['description']} with score {scores[0][0]:.4f}") 
        return result
    else:
        return None

 # No relevant event found
result = retrieve_event_memory("When was our first trip together?")
if result:
    print("Matched Event:", result["event"])
    print("Details:", result["date"], "-", result["description"])
else:
    print("No event matched.")
