from pathlib import Path
import pickle
import re

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


# ============================================================
# PATHS
# ============================================================

RAG_ROOT = Path(
    r"C:\Users\HP\Documents\Weather-Intelligence-AI\api\rag"
)

DOCUMENT_PATH = (
    RAG_ROOT
    / "documents"
    / "weather_knowledge_base.txt"
)

VECTOR_STORE_DIR = (
    RAG_ROOT
    / "vector_store"
)

# New semantic vector store files
SEMANTIC_INDEX_PATH = (
    VECTOR_STORE_DIR
    / "semantic_faiss.index"
)

SEMANTIC_CHUNKS_PATH = (
    VECTOR_STORE_DIR
    / "semantic_chunks.pkl"
)

EMBEDDING_MODEL_PATH = (
    VECTOR_STORE_DIR
    / "embedding_model.txt"
)


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

print(
    f"Document loaded: {DOCUMENT_PATH}"
)


# ============================================================
# CREATE KNOWLEDGE CHUNKS
# ============================================================

def create_chunks(text):

    text = text.replace(
        "\r\n",
        "\n"
    )

    section_pattern = (
        r"(?m)(?=^\d+\.\s+)"
    )

    sections = re.split(
        section_pattern,
        text
    )

    chunks = []

    for section in sections:

        section = section.strip()

        if not section:
            continue

        if not re.search(
            r"^\d+\.\s+",
            section
        ):
            continue

        chunks.append(section)

    # --------------------------------------------------------
    # PROJECT-SPECIFIC CHUNK
    # --------------------------------------------------------

    project_marker = (
        "PROJECT-SPECIFIC WEATHER AI MODEL INFORMATION"
    )

    if project_marker in text:

        project_section = text[
            text.index(project_marker):
        ].strip()

        project_section = (
            "PROJECT-SPECIFIC WEATHER AI MODEL INFORMATION\n\n"
            +
            project_section.split(
                project_marker,
                1
            )[1].strip()
        )

        chunks.append(
            project_section
        )

    return chunks


chunks = create_chunks(text)


# ============================================================
# CHUNK VERIFICATION
# ============================================================

print()
print(
    f"Number of chunks created: {len(chunks)}"
)

print()
print("Chunk verification:")

for i, chunk in enumerate(
    chunks,
    start=1
):

    first_line = (
        chunk.splitlines()[0]
    )

    print(
        f"Chunk {i}: {first_line}"
    )


# ============================================================
# LOAD SEMANTIC EMBEDDING MODEL
# ============================================================

print()
print(
    "Loading semantic embedding model..."
)

MODEL_NAME = "all-MiniLM-L6-v2"

model = SentenceTransformer(
    MODEL_NAME
)

print(
    f"Embedding model: {MODEL_NAME}"
)


# ============================================================
# GENERATE SEMANTIC EMBEDDINGS
# ============================================================

print()
print(
    "Generating semantic embeddings..."
)

embeddings = model.encode(
    chunks,
    convert_to_numpy=True,
    show_progress_bar=True
)

embeddings = embeddings.astype(
    np.float32
)

print(
    f"Embedding shape: {embeddings.shape}"
)


# ============================================================
# NORMALIZE EMBEDDINGS
# ============================================================

faiss.normalize_L2(
    embeddings
)


# ============================================================
# CREATE FAISS VECTOR INDEX
# ============================================================

dimension = embeddings.shape[1]

index = faiss.IndexFlatIP(
    dimension
)

index.add(
    embeddings
)


# ============================================================
# SAVE SEMANTIC VECTOR STORE
# ============================================================

faiss.write_index(
    index,
    str(SEMANTIC_INDEX_PATH)
)

with open(
    SEMANTIC_CHUNKS_PATH,
    "wb"
) as file:

    pickle.dump(
        chunks,
        file
    )

EMBEDDING_MODEL_PATH.write_text(
    MODEL_NAME,
    encoding="utf-8"
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print()
print("=" * 60)
print(
    "SEMANTIC RAG VECTOR STORE CREATED SUCCESSFULLY"
)
print("=" * 60)

print(
    f"Chunks              : {len(chunks)}"
)

print(
    f"Embedding dimension : {dimension}"
)

print(
    f"Embedding model     : {MODEL_NAME}"
)

print(
    f"FAISS index         : {SEMANTIC_INDEX_PATH}"
)

print(
    f"Chunks file         : {SEMANTIC_CHUNKS_PATH}"
)

print(
    f"Model info          : {EMBEDDING_MODEL_PATH}"
)

print("=" * 60)