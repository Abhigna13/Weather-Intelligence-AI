from pathlib import Path
import pickle
import faiss

from sentence_transformers import SentenceTransformer
from langchain_ollama import OllamaLLM


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

VECTOR_STORE_DIR = BASE_DIR / "api" / "rag" / "vector_store"

FAISS_INDEX_PATH = VECTOR_STORE_DIR / "semantic_faiss.index"
CHUNKS_PATH = VECTOR_STORE_DIR / "semantic_chunks.pkl"
EMBEDDING_MODEL_PATH = VECTOR_STORE_DIR / "embedding_model.txt"


# ============================================================
# CONFIGURATION
# ============================================================

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
LLM_MODEL_NAME = "llama3.2:3b"

TOP_K = 3


# ============================================================
# LOAD SEMANTIC RAG COMPONENTS
# ============================================================

def load_rag_components():
    print("Loading LangChain RAG components...")

    print("Loading embedding model...")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    print("Loading semantic FAISS vector store...")
    index = faiss.read_index(str(FAISS_INDEX_PATH))

    print("Loading knowledge chunks...")
    with open(CHUNKS_PATH, "rb") as file:
        chunks = pickle.load(file)

    print("Loading LangChain Ollama LLM...")
    llm = OllamaLLM(
        model=LLM_MODEL_NAME
    )

    print("✓ Embedding model:", EMBEDDING_MODEL_NAME)
    print("✓ Semantic FAISS vector store loaded")
    print("✓ Knowledge chunks:", len(chunks))
    print("✓ Embedding dimension:", index.d)
    print("✓ LangChain Ollama LLM:", LLM_MODEL_NAME)

    return embedding_model, index, chunks, llm


# ============================================================
# SEMANTIC RETRIEVAL
# ============================================================

def retrieve_context(
    question,
    embedding_model,
    index,
    chunks,
    top_k=TOP_K
):
    query_embedding = embedding_model.encode(
        [question],
        normalize_embeddings=True
    )

    scores, indices = index.search(
        query_embedding,
        top_k
    )

    retrieved_chunks = []

    print("\nLANGCHAIN SEMANTIC RETRIEVAL")
    print("-" * 60)

    for rank, (score, index_id) in enumerate(
        zip(scores[0], indices[0]),
        start=1
    ):
        if index_id < 0:
            continue

        chunk = chunks[index_id]

        print(
            f"Result {rank} | "
            f"Semantic Similarity Score: {score:.4f}"
        )

        retrieved_chunks.append(chunk)

    return retrieved_chunks


# ============================================================
# LANGCHAIN RAG PROMPT
# ============================================================

def build_prompt(question, retrieved_chunks):

    context = "\n\n".join(retrieved_chunks)

    prompt = f"""
You are Weather Intelligence AI, a research assistant for a weather
prediction and climate decision support project.

Answer the user's question using the provided project knowledge.

PROJECT KNOWLEDGE:
{context}

USER QUESTION:
{question}

INSTRUCTIONS:
- Answer using the provided project knowledge.
- Be accurate and concise.
- Do not invent project details.
- If the answer is available in the project knowledge, state it clearly.
- If the project knowledge does not contain the answer, say that the
  information is not available in the project knowledge.

ANSWER:
"""

    return prompt


# ============================================================
# LANGCHAIN LLM GENERATION
# ============================================================

def generate_answer(
    question,
    retrieved_chunks,
    llm
):
    prompt = build_prompt(
        question,
        retrieved_chunks
    )

    print(
        "\nGenerating answer using "
        "LangChain + Llama 3.2 3B..."
    )

    answer = llm.invoke(prompt)

    return answer.strip()


# ============================================================
# COMPLETE LANGCHAIN RAG PIPELINE
# ============================================================

def ask_weather_ai(question):

    embedding_model, index, chunks, llm = load_rag_components()

    retrieved_chunks = retrieve_context(
        question,
        embedding_model,
        index,
        chunks
    )

    answer = generate_answer(
        question,
        retrieved_chunks,
        llm
    )

    return answer


# ============================================================
# RESEARCH ASSISTANT INTERFACE
# ============================================================

def langchain_research(question):
    """
    LangChain-based research interface for the
    Weather Research Assistant.
    """

    return ask_weather_ai(question)


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    question = input(
        "\nAsk Weather Intelligence AI: "
    ).strip()

    if not question:
        print("Please enter a question.")

    else:

        print("\n" + "=" * 60)
        print("WEATHER INTELLIGENCE AI — LANGCHAIN RAG")
        print("=" * 60)

        print("\nUser Question:", question)

        answer = ask_weather_ai(question)

        print("\nLANGCHAIN RAG ANSWER")
        print("-" * 60)
        print(answer)
        print("-" * 60)