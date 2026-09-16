print("STARTING LLAMAINDEX RAG")

from pathlib import Path
from typing import List

from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    Settings,
)
from llama_index.core.embeddings import BaseEmbedding
from llama_index.llms.ollama import Ollama

from sentence_transformers import SentenceTransformer


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

KNOWLEDGE_BASE_FILE = (
    PROJECT_ROOT
    / "api"
    / "rag"
    / "documents"
    / "weather_knowledge_base.txt"
)

print("PROJECT ROOT:", PROJECT_ROOT)
print("KNOWLEDGE BASE:", KNOWLEDGE_BASE_FILE)


# ============================================================
# CUSTOM SENTENCE TRANSFORMER EMBEDDING
# ============================================================

class SentenceTransformerEmbedding(BaseEmbedding):
    """
    LlamaIndex-compatible embedding adapter using the
    already-working SentenceTransformer model.
    """

    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(self, model_name: str = model_name, **kwargs):
        super().__init__(**kwargs)

        print("Loading SentenceTransformer...")
        self._model = SentenceTransformer(model_name)

        print("SentenceTransformer loaded successfully.")
        print(
            "Embedding dimension:",
            self._model.get_sentence_embedding_dimension()
        )

    @classmethod
    def class_name(cls) -> str:
        return "SentenceTransformerEmbedding"

    def _get_query_embedding(self, query: str) -> List[float]:
        embedding = self._model.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        return embedding.tolist()

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._get_query_embedding(query)

    def _get_text_embedding(self, text: str) -> List[float]:
        embedding = self._model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        return embedding.tolist()

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        embeddings = self._model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        return embeddings.tolist()


# ============================================================
# CHECK KNOWLEDGE BASE
# ============================================================

if not KNOWLEDGE_BASE_FILE.exists():
    raise FileNotFoundError(
        f"Knowledge base not found:\n{KNOWLEDGE_BASE_FILE}"
    )

print("Knowledge base found.")


# ============================================================
# CONFIGURE OLLAMA
# ============================================================

print("Configuring Ollama...")

Settings.llm = Ollama(
    model="llama3.2:3b",
    request_timeout=600.0,
    context_window=2048,
    temperature=0.1,
)

print("Ollama configured successfully.")


# ============================================================
# CONFIGURE LOCAL EMBEDDINGS
# ============================================================

print("Configuring local SentenceTransformer embeddings...")

Settings.embed_model = SentenceTransformerEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Local embedding model configured successfully.")


# ============================================================
# LOAD KNOWLEDGE BASE
# ============================================================

print("Loading weather knowledge base...")

reader = SimpleDirectoryReader(
    input_files=[str(KNOWLEDGE_BASE_FILE)]
)

documents = reader.load_data()

print("Documents loaded:", len(documents))


# ============================================================
# CREATE LLAMAINDEX VECTOR INDEX
# ============================================================

print("Creating LlamaIndex vector index...")

index = VectorStoreIndex.from_documents(
    documents
)

print("LlamaIndex vector index created successfully.")


# ============================================================
# CREATE QUERY ENGINE
# ============================================================

print("Creating LlamaIndex query engine...")

query_engine = index.as_query_engine(
    similarity_top_k=3
)

print("Query engine created successfully.")


# ============================================================
# TEST RAG QUERY
# ============================================================

QUESTION = (
    "What is RAG and how is it used in this "
    "Weather Intelligence AI project?"
)

print()
print("=" * 70)
print("LLAMAINDEX RAG QUERY")
print("=" * 70)
print(QUESTION)
print("=" * 70)


response = query_engine.query(QUESTION)


# ============================================================
# DISPLAY RESULT
# ============================================================

print()
print("=" * 70)
print("LLAMAINDEX RAG ANSWER")
print("=" * 70)
print(str(response))
print("=" * 70)

print()
print("LLAMAINDEX RAG TEST COMPLETED SUCCESSFULLY")
print("=" * 70)