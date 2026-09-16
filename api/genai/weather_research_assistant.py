import streamlit as st
import sys
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from api.rag.rag_assistant import retrieve_context, generate_llm_answer


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Weather Intelligence AI",
    page_icon="🌦️",
    layout="wide"
)


# ============================================================
# HEADER
# ============================================================

st.title("🌦️ Weather Intelligence AI")
st.subheader("GenAI Weather Research Assistant")

st.write(
    "Ask questions about weather prediction, machine learning, "
    "climate intelligence, satellite analysis, and the Indian Southwest Monsoon."
)

st.divider()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("🤖 AI Assistant")

    st.success("RAG System: Online")
    st.success("Llama 3.2 3B: Ready")
    st.success("FAISS Vector Store: Ready")

    st.divider()

    st.caption("Knowledge Base")
    st.write("Weather Intelligence Knowledge Base")

    st.caption("Retrieval")
    st.write("TF-IDF + FAISS")

    st.caption("Language Model")
    st.write("Llama 3.2 3B via Ollama")


# ============================================================
# QUESTION INPUT
# ============================================================

question = st.text_input(
    "Ask the Weather AI Assistant",
    placeholder="Example: What is Random Forest and how is it used for weather prediction?"
)


# ============================================================
# ASK BUTTON
# ============================================================

if st.button("🔍 Ask Weather AI", type="primary"):

    if not question.strip():

        st.warning("Please enter a question.")

    else:

        with st.spinner("Retrieving knowledge and generating answer..."):

            # Retrieve relevant knowledge
            context = retrieve_context(
                question,
                top_k=3
            )

            if not context:

                st.error(
                    "No relevant information was found in the Weather Knowledge Base."
                )

            else:

                # Generate answer using Llama
                answer = generate_llm_answer(
                    question,
                    context
                )

                # ====================================================
                # ANSWER
                # ====================================================

                st.subheader("🤖 AI Answer")

                if answer:

                    st.success(answer)

                else:

                    st.error(
                        "Unable to generate an answer using the local LLM."
                    )


                # ====================================================
                # RETRIEVED KNOWLEDGE
                # ====================================================

                with st.expander("📚 Retrieved Knowledge"):

                    for i, item in enumerate(context, start=1):

                        st.markdown(
                            f"### Knowledge Result {i}"
                        )

                        st.write(item)

                        st.divider()


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Weather Intelligence AI • RAG + FAISS + Llama 3.2 3B"
)