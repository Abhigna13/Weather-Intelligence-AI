# ============================================================
# SCIENTIFIC LITERATURE RETRIEVAL WORKFLOW
# Weather Intelligence and Climate Decision Support Platform
# Part 02 - Step 2
# ============================================================

from pathlib import Path
import json
import re

from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.core.embeddings import BaseEmbedding
from llama_index.llms.ollama import Ollama
from sentence_transformers import SentenceTransformer


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

OUTPUT_DIR = PROJECT_ROOT / "reports" / "literature"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LITERATURE_FILE = OUTPUT_DIR / "scientific_literature.json"
RETRIEVAL_FILE = OUTPUT_DIR / "literature_retrieval_result.json"


# ============================================================
# LOCAL EMBEDDING ADAPTER
# ============================================================

class SentenceTransformerEmbedding(BaseEmbedding):

    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(
        self,
        model_name: str = model_name,
        **kwargs
    ):
        super().__init__(**kwargs)

        print("Loading local embedding model...")

        self._model = SentenceTransformer(model_name)

        print("Embedding model loaded.")
        print(
            "Embedding dimension:",
            self._model.get_embedding_dimension()
        )

    @classmethod
    def class_name(cls) -> str:
        return "SentenceTransformerEmbedding"

    def _get_query_embedding(self, query: str):
        embedding = self._model.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        return embedding.tolist()

    async def _aget_query_embedding(self, query: str):
        return self._get_query_embedding(query)

    def _get_text_embedding(self, text: str):
        embedding = self._model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        return embedding.tolist()

    def _get_text_embeddings(self, texts):
        embeddings = self._model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        return embeddings.tolist()


# ============================================================
# SCIENTIFIC LITERATURE KNOWLEDGE BASE
# ============================================================

LITERATURE = [
    {
        "id": "LIT-001",
        "title": "Machine Learning for Weather and Climate Prediction",
        "domain": "Machine Learning",
        "keywords": [
            "machine learning",
            "weather prediction",
            "climate prediction",
            "forecasting"
        ],
        "summary": (
            "Machine learning methods can learn nonlinear relationships "
            "between meteorological variables and weather outcomes. "
            "Tree-based ensemble models are particularly useful for "
            "structured meteorological datasets."
        )
    },
    {
        "id": "LIT-002",
        "title": "Deep Learning for Weather Forecasting",
        "domain": "Deep Learning",
        "keywords": [
            "deep learning",
            "LSTM",
            "GRU",
            "weather forecasting"
        ],
        "summary": (
            "Deep learning models can represent complex temporal "
            "relationships in meteorological observations. Recurrent "
            "architectures such as LSTM and GRU can be applied to "
            "sequential weather data."
        )
    },
    {
        "id": "LIT-003",
        "title": "Time-Series Forecasting for Meteorological Data",
        "domain": "Time-Series Forecasting",
        "keywords": [
            "ARIMA",
            "SARIMA",
            "time series",
            "temperature forecasting"
        ],
        "summary": (
            "Statistical time-series models such as ARIMA and SARIMA "
            "can represent temporal dependencies and seasonal patterns "
            "in meteorological variables."
        )
    },
    {
        "id": "LIT-004",
        "title": "Remote Sensing and Satellite-Based Weather Analysis",
        "domain": "Remote Sensing",
        "keywords": [
            "remote sensing",
            "satellite",
            "cloud detection",
            "computer vision"
        ],
        "summary": (
            "Satellite imagery provides spatial information that can "
            "complement ground-based meteorological observations. "
            "Computer vision techniques can classify cloud-related "
            "conditions from satellite images."
        )
    },
    {
        "id": "LIT-005",
        "title": "Explainable Artificial Intelligence for Weather Prediction",
        "domain": "Explainable AI",
        "keywords": [
            "XAI",
            "SHAP",
            "explainability",
            "weather prediction"
        ],
        "summary": (
            "Explainable AI methods help identify how input variables "
            "contribute to machine learning predictions. SHAP provides "
            "feature-level contribution values for model interpretation."
        )
    },
    {
        "id": "LIT-006",
        "title": "Retrieval-Augmented Generation for Scientific Knowledge",
        "domain": "Generative AI",
        "keywords": [
            "RAG",
            "retrieval augmented generation",
            "LLM",
            "scientific knowledge"
        ],
        "summary": (
            "Retrieval-Augmented Generation combines document retrieval "
            "with language generation. Relevant scientific information "
            "can be retrieved before an LLM generates a grounded response."
        )
    },
    {
        "id": "LIT-007",
        "title": "Large Language Models for Scientific Research Assistance",
        "domain": "Large Language Models",
        "keywords": [
            "LLM",
            "large language models",
            "research assistant",
            "scientific research"
        ],
        "summary": (
            "Large language models can support scientific research "
            "through question answering, summarization, information "
            "retrieval and structured research workflows."
        )
    },
    {
        "id": "LIT-008",
        "title": "AI Agents for Analytical Weather Workflows",
        "domain": "AI Agents",
        "keywords": [
            "AI agents",
            "tool calling",
            "weather analytics",
            "workflow"
        ],
        "summary": (
            "AI agents can coordinate multiple tools and analytical "
            "operations. Tool-based workflows allow an agent to route "
            "questions to appropriate prediction, retrieval or analysis "
            "components."
        )
    }
]


# ============================================================
# SAVE LITERATURE KNOWLEDGE BASE
# ============================================================

print("Preparing scientific literature knowledge base...")

with open(
    LITERATURE_FILE,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        LITERATURE,
        file,
        indent=4,
        ensure_ascii=False
    )

print("Literature knowledge base saved:")
print(LITERATURE_FILE)


# ============================================================
# CONFIGURE LOCAL LLM
# ============================================================

print("Configuring local Llama 3.2...")

Settings.llm = Ollama(
    model="llama3.2:3b",
    request_timeout=600.0,
    context_window=2048,
    temperature=0.1
)

print("Llama 3.2 configured.")


# ============================================================
# CONFIGURE LOCAL EMBEDDINGS
# ============================================================

Settings.embed_model = SentenceTransformerEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Local embeddings configured.")


# ============================================================
# CONVERT LITERATURE INTO LLAMAINDEX DOCUMENTS
# ============================================================

print("Creating literature documents...")

documents = []

for item in LITERATURE:

    text = (
        f"Title: {item['title']}\n"
        f"Domain: {item['domain']}\n"
        f"Keywords: {', '.join(item['keywords'])}\n"
        f"Summary: {item['summary']}"
    )

    documents.append(
        Document(
            text=text,
            metadata={
                "literature_id": item["id"],
                "title": item["title"],
                "domain": item["domain"]
            }
        )
    )

print("Literature documents:", len(documents))


# ============================================================
# CREATE VECTOR INDEX
# ============================================================

print("Creating scientific literature vector index...")

index = VectorStoreIndex.from_documents(
    documents
)

print("Literature vector index created.")


# ============================================================
# CREATE QUERY ENGINE
# ============================================================

query_engine = index.as_query_engine(
    similarity_top_k=3
)

print("Literature query engine created.")


# ============================================================
# RESEARCH QUESTION
# ============================================================

RESEARCH_QUESTION = (
    "What machine learning and deep learning approaches are "
    "useful for weather prediction and climate intelligence?"
)


# ============================================================
# RETRIEVE LITERATURE
# ============================================================

print()
print("=" * 70)
print("SCIENTIFIC LITERATURE RETRIEVAL")
print("=" * 70)

print("Research question:")
print(RESEARCH_QUESTION)

print("=" * 70)

retriever = index.as_retriever(
    similarity_top_k=3
)

retrieved_nodes = retriever.retrieve(
    RESEARCH_QUESTION
)

print()
print("Retrieved literature sources:", len(retrieved_nodes))


retrieved_sources = []

for rank, node in enumerate(
    retrieved_nodes,
    start=1
):

    metadata = node.node.metadata

    source = {
        "rank": rank,
        "literature_id": metadata.get(
            "literature_id",
            "unknown"
        ),
        "title": metadata.get(
            "title",
            "unknown"
        ),
        "domain": metadata.get(
            "domain",
            "unknown"
        ),
        "similarity_score": float(
            node.score or 0.0
        )
    }

    retrieved_sources.append(source)

    print()
    print(f"Rank {rank}")
    print("Title:", source["title"])
    print("Domain:", source["domain"])
    print(
        "Similarity:",
        round(source["similarity_score"], 4)
    )


# ============================================================
# GENERATE SCIENTIFIC SUMMARY
# ============================================================

print()
print("Generating research summary with Llama 3.2...")

response = query_engine.query(
    RESEARCH_QUESTION
)

research_summary = str(response)


# ============================================================
# SAVE RETRIEVAL RESULT
# ============================================================

result = {
    "success": True,
    "workflow": (
        "Scientific Literature Retrieval "
        "using LlamaIndex and Llama 3.2"
    ),
    "research_question": RESEARCH_QUESTION,
    "retrieved_sources": retrieved_sources,
    "generated_summary": research_summary
}


with open(
    RETRIEVAL_FILE,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        result,
        file,
        indent=4,
        ensure_ascii=False
    )


# ============================================================
# FINAL OUTPUT
# ============================================================

print()
print("=" * 70)
print("SCIENTIFIC RESEARCH SUMMARY")
print("=" * 70)

print(research_summary)

print("=" * 70)
print("Saved retrieval result:")
print(RETRIEVAL_FILE)

print()
print("=" * 70)
print("SCIENTIFIC LITERATURE RETRIEVAL COMPLETED SUCCESSFULLY")
print("=" * 70)