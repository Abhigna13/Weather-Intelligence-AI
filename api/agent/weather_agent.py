import sys
from pathlib import Path
import requests


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# IMPORT RAG COMPONENTS
# ============================================================

from api.rag.rag_assistant import (
    retrieve_context,
    generate_llm_answer
)


# ============================================================
# IMPORT AGENT MEMORY
# ============================================================

from api.agent.memory import (
    get_recent_memory,
    save_memory
)


# ============================================================
# FASTAPI CONFIGURATION
# ============================================================

API_URL = "http://127.0.0.1:8001/predict"


# ============================================================
# QUERY ROUTER
# ============================================================

def classify_query(query):

    query = query.lower().strip()

    # --------------------------------------------------------
    # MEMORY QUESTIONS
    # --------------------------------------------------------

    memory_keywords = [
        "mentioned earlier",
        "mentioned before",
        "said earlier",
        "said before",
        "told me earlier",
        "told me before",
        "you mentioned",
        "you said",
        "earlier",
        "previously",
        "previous answer",
        "previous question",
        "last answer",
        "last question"
    ]

    for keyword in memory_keywords:

        if keyword in query:
            return "rag"


    # --------------------------------------------------------
    # MULTI-TOOL QUESTIONS
    # --------------------------------------------------------

    multi_tool_keywords = [
        "predict and explain",
        "predict and tell",
        "predict and describe",
        "prediction and explain",
        "temperature and explain",
        "predict the temperature and explain",
        "predict temperature and explain"
    ]

    for keyword in multi_tool_keywords:

        if keyword in query:
            return "multi_tool"


    # --------------------------------------------------------
    # RESEARCH / KNOWLEDGE QUESTIONS
    # --------------------------------------------------------

    research_keywords = [
        "what is",
        "how is",
        "explain",
        "define",
        "meaning of",
        "which model",
        "what model",
        "random forest",
        "decision tree",
        "linear regression",
        "gradient boosting",
        "lstm",
        "gru",
        "cnn-lstm",
        "arima",
        "sarima",
        "rag",
        "machine learning",
        "deep learning",
        "time series",
        "weather model",
        "weather prediction model"
    ]

    for keyword in research_keywords:

        if keyword in query:
            return "rag"


    # --------------------------------------------------------
    # PREDICTION QUESTIONS
    # --------------------------------------------------------

    prediction_keywords = [
        "predict temperature",
        "temperature prediction",
        "predict the temperature",
        "what will be the temperature",
        "forecast temperature",
        "give me the temperature",
        "temperature tomorrow",
        "temperature today"
    ]

    for keyword in prediction_keywords:

        if keyword in query:
            return "prediction"


    # --------------------------------------------------------
    # DEFAULT
    # --------------------------------------------------------

    return "rag"


# ============================================================
# WEATHER PREDICTION TOOL
# ============================================================

def predict_temperature(values):

    if len(values) != 13:

        return {
            "success": False,
            "error": "Exactly 13 weather parameters are required."
        }


    try:

        response = requests.post(
            API_URL,
            json={
                "values": values
            },
            timeout=10
        )

        response.raise_for_status()

        result = response.json()

        return result


    except requests.exceptions.ConnectionError:

        return {
            "success": False,
            "error": "Weather FastAPI server is not running."
        }


    except requests.exceptions.Timeout:

        return {
            "success": False,
            "error": "Weather prediction API request timed out."
        }


    except Exception as error:

        return {
            "success": False,
            "error": str(error)
        }


# ============================================================
# RAG WEATHER RESEARCH TOOL
# ============================================================

def research_weather(query):

    # --------------------------------------------------------
    # LOAD RECENT MEMORY
    # --------------------------------------------------------

    recent_memory = get_recent_memory(
        limit=5
    )

    query_lower = query.lower().strip()


    # --------------------------------------------------------
    # DIRECT PROJECT MODEL ANSWER
    # --------------------------------------------------------

    model_keywords = [
        "which machine learning model",
        "what machine learning model",
        "which model is used",
        "what model is used",
        "model used for temperature prediction",
        "temperature prediction model",
        "model for temperature prediction"
    ]

    if any(
        keyword in query_lower
        for keyword in model_keywords
    ):

        return {
            "answer": (
                "The primary machine learning model used for "
                "temperature prediction in the deployed Weather "
                "Intelligence AI application is RandomForestRegressor."
            ),
            "source": "Project Knowledge Base",
            "model": "RandomForestRegressor"
        }


    # --------------------------------------------------------
    # DIRECT MEMORY RECALL
    # --------------------------------------------------------

    memory_keywords = [
        "mentioned earlier",
        "mentioned before",
        "said earlier",
        "said before",
        "told me earlier",
        "told me before",
        "you mentioned",
        "you said",
        "earlier",
        "previously",
        "previous answer",
        "last answer"
    ]

    is_memory_question = any(
        keyword in query_lower
        for keyword in memory_keywords
    )


    if is_memory_question and recent_memory:

        for item in reversed(recent_memory):

            previous_response = item.get(
                "response",
                ""
            )

            if not isinstance(
                previous_response,
                dict
            ):
                continue


            model_info = previous_response.get(
                "model"
            )

            if isinstance(
                model_info,
                dict
            ):

                model_name = model_info.get(
                    "model_name"
                )

                if model_name:

                    return {
                        "answer":
                        f"The model mentioned earlier was {model_name}."
                    }


            answer = previous_response.get(
                "answer"
            )

            if answer:

                answer_text = str(
                    answer
                )

                invalid_answers = [
                    "RMS error model",
                    "RMS-Error model",
                    "conversation just started",
                    "haven't discussed"
                ]

                if not any(
                    invalid in answer_text
                    for invalid in invalid_answers
                ):

                    return {
                        "answer": answer_text
                    }


    # --------------------------------------------------------
    # RAG RETRIEVAL
    # --------------------------------------------------------

    context = retrieve_context(
        query,
        top_k=3
    )

    if not context:

        return {
            "answer": (
                "No relevant information was found in the "
                "Weather Knowledge Base."
            )
        }


    # --------------------------------------------------------
    # COMBINE RAG + MEMORY
    # --------------------------------------------------------

    memory_text = ""

    if recent_memory:

        memory_text = (
            "\n\nPrevious Agent Interactions:\n"
        )

        for item in recent_memory:

            memory_text += (
                f"User: "
                f"{item.get('user_query', '')}\n"
                f"Agent: "
                f"{item.get('response', '')}\n\n"
            )


    context_with_memory = list(
        context
    )

    if memory_text:

        context_with_memory.append(
            memory_text
        )


    # --------------------------------------------------------
    # GENERATE LLM ANSWER
    # --------------------------------------------------------

    answer = generate_llm_answer(
        query,
        context_with_memory
    )

    if answer is None:

        return {
            "answer": (
                "Unable to generate an answer using the local LLM."
            )
        }


    return {
        "answer": answer
    }

# ============================================================
# TOOL REGISTRY
# ============================================================

TOOLS = {

    "prediction": {
        "name": "Weather Prediction Tool",
        "function": predict_temperature
    },

    "research": {
        "name": "Weather Research Tool",
        "function": research_weather
    }

}


# ============================================================
# AI AGENT
# ============================================================

def run_weather_agent(
    query,
    values=None
):

    print("\n" + "=" * 60)

    print(
        "WEATHER INTELLIGENCE AI AGENT"
    )

    print("=" * 60)

    print(
        f"\nUser Query : {query}"
    )


    # --------------------------------------------------------
    # SHOW RECENT MEMORY
    # --------------------------------------------------------

    recent_memory = get_recent_memory(
        limit=5
    )

    if recent_memory:

        print(
            "\nRecent Agent Memory:"
        )

        print(
            "-" * 60
        )

        for item in recent_memory:

            print(
                f"User : "
                f"{item.get('user_query', '')}"
            )

            print(
                f"Agent: "
                f"{item.get('response', '')}"
            )

    else:

        print(
            "\nRecent Agent Memory: None"
        )


    # --------------------------------------------------------
    # QUERY CLASSIFICATION
    # --------------------------------------------------------

    route = classify_query(
        query
    )

    print(
        f"\nSelected Tool : {route}"
    )


    # ========================================================
    # MULTI-TOOL WORKFLOW
    # ========================================================

    if route == "multi_tool":

        print(
            "\nExecuting Multi-Tool Workflow..."
        )


        # ----------------------------------------------------
        # TOOL 1 — PREDICTION
        # ----------------------------------------------------

        print(
            "\n[Tool 1] Prediction Tool"
        )

        print(
            "-" * 60
        )


        if values is None:

            values = [
                19,
                0,
                18,
                0.65,
                14,
                225,
                12,
                2,
                1008,
                150,
                2026,
                9,
                11
            ]


        prediction_result = TOOLS[
            "prediction"
        ][
            "function"
        ](
            values
        )


        print(
            "Prediction Tool Result:"
        )

        print(
            prediction_result
        )


        # ----------------------------------------------------
        # TOOL 2 — RESEARCH
        # ----------------------------------------------------

        print(
            "\n[Tool 2] Research Tool"
        )

        print(
            "-" * 60
        )


        research_query = (
            "Which machine learning model is used "
            "for temperature prediction in this project?"
        )


        research_result = TOOLS[
            "research"
        ][
            "function"
        ](
            research_query
        )


        print(
            "Research Tool Result:"
        )

        print(
            research_result
        )


        # ----------------------------------------------------
        # COMBINED RESULT
        # ----------------------------------------------------

        combined_result = {

            "success": True,

            "workflow": "multi_tool",

            "prediction": prediction_result,

            "research": research_result

        }


        print(
            "\nCOMBINED MULTI-TOOL RESULT"
        )

        print(
            "-" * 60
        )

        print(
            combined_result
        )


        # ----------------------------------------------------
        # SAVE MEMORY
        # ----------------------------------------------------

        save_memory(
            query,
            combined_result
        )


        print(
            "\n✓ Multi-tool interaction "
            "saved to agent memory."
        )


        return combined_result


    # ========================================================
    # PREDICTION WORKFLOW
    # ========================================================

    if route == "prediction":

        print(
            "\nRouting to Weather Prediction API..."
        )


        if values is None:

            values = [
                19,
                0,
                18,
                0.65,
                14,
                225,
                12,
                2,
                1008,
                150,
                2026,
                9,
                11
            ]


        result = TOOLS[
            "prediction"
        ][
            "function"
        ](
            values
        )


        print(
            "\nPREDICTION RESULT"
        )

        print(
            "-" * 60
        )

        print(
            result
        )


        save_memory(
            query,
            result
        )


        print(
            "\n✓ Prediction interaction "
            "saved to agent memory."
        )


        return result


    # ========================================================
    # RAG WORKFLOW
    # ========================================================

    print(
        "\nRouting to RAG Knowledge Base..."
    )


    result = TOOLS[
        "research"
    ][
        "function"
    ](
        query
    )


    print(
        "\nRAG + LLM RESULT"
    )

    print(
        "-" * 60
    )

    print(
        result["answer"]
    )


    save_memory(
        query,
        result
    )


    print(
        "\n✓ Research interaction "
        "saved to agent memory."
    )


    return result


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    question = input(
        "\nAsk the Weather AI Agent: "
    ).strip()


    if question:

        run_weather_agent(
            question
        )

    else:

        print(
            "Please enter a question."
        )