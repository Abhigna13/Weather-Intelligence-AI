from pathlib import Path
import pickle
import faiss
import numpy as np


RAG_ROOT = Path(r"C:\Users\HP\Documents\Weather-Intelligence-AI\api\rag")

VECTOR_STORE_DIR = RAG_ROOT / "vector_store"

INDEX_PATH = VECTOR_STORE_DIR / "weather_faiss.index"
CHUNKS_PATH = VECTOR_STORE_DIR / "weather_chunks.pkl"
VECTORIZER_PATH = VECTOR_STORE_DIR / "tfidf_vectorizer.pkl"


print("Loading RAG vector store...")

index = faiss.read_index(str(INDEX_PATH))

with open(CHUNKS_PATH, "rb") as f:
    chunks = pickle.load(f)

with open(VECTORIZER_PATH, "rb") as f:
    vectorizer = pickle.load(f)

print("Vector store loaded successfully.")
print(f"Total chunks: {len(chunks)}")
print(f"Vector size: {index.d}")
print()


query = input("Enter your weather question: ")

query_vector = vectorizer.transform([query]).toarray().astype("float32")

faiss.normalize_L2(query_vector)

scores, indices = index.search(query_vector, 3)


print()
print("=" * 70)
print("TOP RETRIEVED WEATHER KNOWLEDGE")
print("=" * 70)

for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):

    print()
    print(f"Result {rank}")
    print(f"Similarity Score: {score:.4f}")
    print("-" * 70)
    print(chunks[idx])

print()
print("=" * 70)
print("RETRIEVAL TEST COMPLETED")
print("=" * 70)
