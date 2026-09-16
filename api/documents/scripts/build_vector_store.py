from pathlib import Path
import pickle
import re

import faiss
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


# ============================================================
# RAG PATHS
# ============================================================

RAG_ROOT = Path(
    r"C:\Users\HP\Documents\Weather-Intelligence-AI\api\rag"
)

DOCUMENT_PATH = (
    RAG_ROOT
    / "documents"
    / "weather_knowledge_base.txt"
)

VECTOR_STORE_DIR = RAG_ROOT / "vector_store"

INDEX_PATH = VECTOR_STORE_DIR / "weather_faiss.index"
CHUNKS_PATH = VECTOR_STORE_DIR / "weather_chunks.pkl"
VECTORIZER_PATH = VECTOR_STORE_DIR / "tfidf_vectorizer.pkl"


# ============================================================
# CREATE VECTOR STORE DIRECTORY
# ============================================================

VECTOR_STORE_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD KNOWLEDGE BASE
# ============================================================

print("Loading Weather Knowledge Base...")

if not DOCUMENT_PATH.exists():
    raise FileNotFoundError(
        f"Knowledge base not found:\n{DOCUMENT_PATH}"
    )

text = DOCUMENT_PATH.read_text(
    encoding="utf-8"
)

print(f"Document loaded: {DOCUMENT_PATH}")


# ============================================================
# SECTION-BASED CHUNKING
# ============================================================

def create_chunks(text):

    # Normalize line endings
    text = text.replace("\r\n", "\n")

    # Split numbered sections such as:
    # 1. TEMPERATURE
    # 2. HUMIDITY
    # ...
    # 20. SATELLITE WEATHER ANALYSIS

    section_pattern = r"(?m)(?=^\d+\.\s+)"

    sections = re.split(
        section_pattern,
        text
    )

    chunks = []

    for section in sections:

        section = section.strip()

        if not section:
            continue

        # Ignore the initial title-only portion
        if not re.search(
            r"^\d+\.\s+",
            section
        ):
            continue

        chunks.append(section)

    # --------------------------------------------------------
    # Add project-specific information as a dedicated chunk
    # --------------------------------------------------------

    project_marker = (
        "PROJECT-SPECIFIC WEATHER AI MODEL INFORMATION"
    )

    if project_marker in text:

        project_section = text[
            text.index(
                project_marker
            ):
        ].strip()

        # Include the separator/header in the chunk
        project_section = (
            "PROJECT-SPECIFIC WEATHER AI MODEL INFORMATION\n\n"
            + project_section.split(
                project_marker,
                1
            )[1].strip()
        )

        chunks.append(project_section)

    return chunks


# ============================================================
# CREATE CHUNKS
# ============================================================

chunks = create_chunks(text)

print(
    f"Number of chunks created: {len(chunks)}"
)

print()
print("Chunk verification:")

for i, chunk in enumerate(chunks, start=1):

    first_line = chunk.splitlines()[0]

    print(
        f"Chunk {i}: {first_line}"
    )


# ============================================================
# TF-IDF EMBEDDINGS
# ============================================================

print()
print("Generating lightweight embeddings...")

vectorizer = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1, 2)
)

vectors = vectorizer.fit_transform(
    chunks
)

vectors = vectors.astype(
    np.float32
).toarray()

print(
    f"Embedding shape: {vectors.shape}"
)


# ============================================================
# NORMALIZE VECTORS
# ============================================================

faiss.normalize_L2(
    vectors
)


# ============================================================
# CREATE FAISS INDEX
# ============================================================

dimension = vectors.shape[1]

index = faiss.IndexFlatIP(
    dimension
)

index.add(
    vectors
)


# ============================================================
# SAVE FAISS INDEX
# ============================================================

faiss.write_index(
    index,
    str(INDEX_PATH)
)


# ============================================================
# SAVE CHUNKS
# ============================================================

with open(
    CHUNKS_PATH,
    "wb"
) as file:

    pickle.dump(
        chunks,
        file
    )


# ============================================================
# SAVE TF-IDF VECTORIZER
# ============================================================

with open(
    VECTORIZER_PATH,
    "wb"
) as file:

    pickle.dump(
        vectorizer,
        file
    )


# ============================================================
# FINAL OUTPUT
# ============================================================

print()
print("=" * 60)
print("RAG VECTOR STORE CREATED SUCCESSFULLY")
print("=" * 60)

print(
    f"Chunks       : {len(chunks)}"
)

print(
    f"Vector size  : {dimension}"
)

print(
    f"FAISS index  : {INDEX_PATH}"
)

print(
    f"Chunks file  : {CHUNKS_PATH}"
)

print(
    f"Vectorizer   : {VECTORIZER_PATH}"
)

print("=" * 60)