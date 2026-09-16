from pathlib import Path
import pickle

import faiss
import numpy as np


# RAG root
RAG_ROOT = Path(
    r"C:\Users\HP\Documents\Weather-Intelligence-AI\api\rag"
)

VECTOR_STORE_DIR = RAG_ROOT / "vector_store"

INDEX_PATH = VECTOR_STORE_DIR / "weather_faiss.index"
CHUNKS_PATH = VECTOR_STORE_DIR / "weather_chunks.pkl"
VECTORIZER_PATH = VECTOR_STORE_DIR / "tfidf_vectorizer.pkl"


print("Loading RAG vector store...")

index = faiss.read_index(str(INDEX_PATH))

with open(CHUNKS_PATH, "rb") as file:
    chunks = pickle.load(file)

with open(VECTORIZER_PATH, "rb") as file:
    vectorizer = pickle.load(file)


print(f"FAISS index loaded: {index.ntotal} vectors")
print(f"Knowledge chunks loaded: {len(chunks)}")


# User query
query = input("\nEnter your weather question: ").strip()

if not query:
    print("Please enter a question.")
    raise SystemExit


# Convert query to TF-IDF vector
query_vector = vectorizer.transform([query])

query_vector = query_vector.astype(
    np.float32
).toarray()


# Normalize query vector
faiss.normalize_L2(query_vector)


# Retrieve top results
top_k = min(3, len(chunks))

scores, indices = index.search(
    query_vector,
    top_k
)


print("\n" + "=" * 60)
print("RAG RETRIEVAL RESULTS")
print("=" * 60)

for rank, (score, idx) in enumerate(
    zip(scores[0], indices[0]),
    start=1
):

    print(f"\nResult {rank}")
    print(f"Similarity Score: {score:.4f}")
    print("-" * 60)
    print(chunks[idx])

print("\n" + "=" * 60)
print("RAG RETRIEVAL TEST COMPLETED")
print("=" * 60)