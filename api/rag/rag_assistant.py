from pathlib import Path
import pickle
import requests

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


# ============================================================
# PATHS
# ============================================================

RAG_ROOT = Path(__file__).resolve().parent

VECTOR_STORE_DIR = RAG_ROOT / "vector_store"

# New semantic vector store
SEMANTIC_INDEX_PATH = VECTOR_STORE_DIR / "semantic_faiss.index"
SEMANTIC_CHUNKS_PATH = VECTOR_STORE_DIR / "semantic_chunks.pkl"
EMBEDDING_MODEL_PATH = VECTOR_STORE_DIR / "embedding_model.txt"

# Existing TF-IDF vector store
TFIDF_INDEX_PATH = VECTOR_STORE_DIR / "weather_faiss.index"
TFIDF_CHUNKS_PATH = VECTOR_STORE_DIR / "weather_chunks.pkl"
TFIDF_VECTORIZER_PATH = VECTOR_STORE_DIR / "tfidf_vectorizer.pkl"


# ============================================================
# SEMANTIC EMBEDDING MODEL
# ============================================================

MODEL_NAME = "all-MiniLM-L6-v2"


def load_semantic_model():

    print("Loading semantic embedding model...")

    model = SentenceTransformer(MODEL_NAME)

    print(
        f"✓ Embedding model loaded: {MODEL_NAME}"
    )

    return model


# ============================================================
# LOAD SEMANTIC RAG COMPONENTS
# ============================================================

def load_rag_components():

    print("Loading Semantic RAG Knowledge Base...")

    if not SEMANTIC_INDEX_PATH.exists():
        raise FileNotFoundError(
            f"Semantic FAISS index not found:\n"
            f"{SEMANTIC_INDEX_PATH}"
        )

    if not SEMANTIC_CHUNKS_PATH.exists():
        raise FileNotFoundError(
            f"Semantic chunks file not found:\n"
            f"{SEMANTIC_CHUNKS_PATH}"
        )

    # Load semantic FAISS index
    index = faiss.read_index(
        str(SEMANTIC_INDEX_PATH)
    )

    # Load knowledge chunks
    with open(
        SEMANTIC_CHUNKS_PATH,
        "rb"
    ) as file:

        chunks = pickle.load(file)

    # Load embedding model
    model = load_semantic_model()

    print("✓ Semantic FAISS vector store loaded")
    print(
        f"✓ Knowledge chunks: {len(chunks)}"
    )
    print(
        f"✓ Embedding dimension: {index.d}"
    )

    return index, chunks, model


# ============================================================
# SEMANTIC RETRIEVAL
# ============================================================

def retrieve_context(query, top_k=3):

    index, chunks, model = (
        load_rag_components()
    )

    # Convert user query into semantic embedding
    query_embedding = model.encode(
        [query],
        convert_to_numpy=True
    )

    query_embedding = (
        query_embedding
        .astype(np.float32)
    )

    # Normalize embeddings for cosine similarity
    faiss.normalize_L2(
        query_embedding
    )

    # Search semantic FAISS index
    scores, indices = index.search(
        query_embedding,
        top_k
    )

    retrieved_chunks = []

    print("\nSEMANTIC RAG RETRIEVAL")

    for rank, (score, index_id) in enumerate(
        zip(scores[0], indices[0]),
        start=1
    ):

        if index_id < 0 or index_id >= len(chunks):
            continue

        chunk = chunks[index_id]

        print(
            f"Result {rank} | "
            f"Semantic Similarity Score: "
            f"{score:.4f}"
        )

        retrieved_chunks.append(chunk)

    return retrieved_chunks


# ============================================================
# GENERATE ANSWER USING OLLAMA
# ============================================================

def generate_llm_answer(query, context):

    # Context may contain both RAG knowledge
    # and previous agent memory.
    context_text = "\n\n".join(context)

    # Keep local LLM prompt lightweight.
    context_text = context_text[:10000]

    prompt = f"""
You are Weather Intelligence AI, a weather and climate research assistant.

Answer the user's question using the provided context.

IMPORTANT:
- The context contains Weather Knowledge Base information.
- The context may also contain Previous Agent Interactions.
- Previous Agent Interactions represent real earlier interactions with this AI Agent.
- If the user asks what was mentioned, said, discussed, or answered earlier,
  use the Previous Agent Interactions.
- Do NOT say that the conversation just started when previous interactions
  are provided.
- Do not invent facts.
- If the requested information cannot be found in the provided context,
  clearly say that it is not available.
- Keep the answer concise and accurate.

PROVIDED CONTEXT:
============================================================
{context_text}
============================================================

USER QUESTION:
{query}

ANSWER:
"""

    try:

        response = requests.post(
            "http://127.0.0.1:11434/api/generate",
            json={
                "model": "llama3.2:3b",
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0,
                    "num_predict": 128
                },
                "keep_alive": "5m"
            },
            timeout=300
        )

        response.raise_for_status()

        data = response.json()

        answer = data.get(
            "response",
            ""
        ).strip()

        if not answer:

            print(
                "\nOllama returned an empty response."
            )

            return None

        return answer

    except requests.exceptions.Timeout:

        print(
            "\nOllama API Error: "
            "Generation timed out."
        )

        return None

    except requests.exceptions.RequestException as error:

        print(
            f"\nOllama API Error: {error}"
        )

        return None

    except Exception as error:

        print(
            f"\nLLM Error: {error}"
        )

        return None


# ============================================================
# MAIN RAG + LLM ASSISTANT
# ============================================================

def ask_weather_ai(query):

    print(
        "\n" + "=" * 60
    )

    print(
        "WEATHER INTELLIGENCE AI"
    )

    print(
        "=" * 60
    )

    print(
        f"\nUser Question: {query}"
    )

    # Retrieve relevant knowledge
    context = retrieve_context(
        query,
        top_k=3
    )

    if not context:

        print(
            "\nNo relevant knowledge found."
        )

        return (
            "I could not find relevant information "
            "in the Weather Knowledge Base."
        )

    # Generate answer using Llama
    print(
        "\nGenerating answer using "
        "Llama 3.2 3B..."
    )

    answer = generate_llm_answer(
        query,
        context
    )

    if answer is None:

        return (
            "Unable to generate an answer "
            "using the local LLM."
        )

    print(
        "\nRAG + LLM ANSWER"
    )

    print(
        "-" * 60
    )

    print(answer)

    print(
        "-" * 60
    )

    return answer


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    question = input(
        "\nAsk Weather Intelligence AI: "
    ).strip()

    if question:

        ask_weather_ai(
            question
        )

    else:

        print(
            "Please enter a question."
        )