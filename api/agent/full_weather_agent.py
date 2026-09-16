import ast

from typing import TypedDict, Optional, List

from langgraph.graph import StateGraph, START, END

from api.rag.langchain_rag import langchain_research
from api.mcp.weather_mcp_client import mcp_predict_temperature


class FullWeatherState(TypedDict):
    question: str
    values: Optional[List[float]]
    route: str
    answer: str


def classify_request(state: FullWeatherState):
    question = state["question"].lower()

    prediction_phrases = [
        "predict temperature",
        "predict weather",
        "make a prediction",
        "give me a prediction",
        "forecast temperature",
    ]

    if any(phrase in question for phrase in prediction_phrases):
        route = "prediction"
    else:
        route = "research"

    return {
        "route": route
    }


def research_node(state: FullWeatherState):
    question = state["question"]

    research_prompt = f"""
You are the Weather Intelligence Research Assistant
for an AI-powered weather and climate decision support platform.

Answer the user's question using the retrieved weather knowledge
provided by the RAG system.

User Question:
{question}

Instructions:
1. Give a clear and technically accurate answer.
2. Use the retrieved knowledge as the primary source.
3. Do not invent scientific facts, datasets, results, or citations.
4. If the retrieved knowledge does not contain enough information,
   clearly state that the available knowledge base is insufficient.
5. Keep the answer focused on the user's question.
6. Explain technical concepts clearly.
7. When relevant, connect the answer to the Weather Intelligence AI project.

Response format:

Answer:
<your answer>

Source:
Weather Intelligence Knowledge Base
"""

    answer = langchain_research(research_prompt)

    return {
        "answer": answer
    }

def prediction_node(state: FullWeatherState):

    values = state.get("values")

    if values is None:
        return {
            "answer": (
                "To predict temperature, please provide "
                "the 13 required weather parameters."
            )
        }

    result = mcp_predict_temperature(values)

    try:
        result = ast.literal_eval(result)
    except (ValueError, SyntaxError):
        pass

    return {
        "answer": result
    }

def route_request(state: FullWeatherState):
    return state["route"]


def build_full_weather_agent():

    graph = StateGraph(FullWeatherState)

    graph.add_node("classify", classify_request)
    graph.add_node("research", research_node)
    graph.add_node("prediction", prediction_node)

    graph.add_edge(START, "classify")

    graph.add_conditional_edges(
        "classify",
        route_request,
        {
            "research": "research",
            "prediction": "prediction",
        },
    )

    graph.add_edge("research", END)
    graph.add_edge("prediction", END)

    return graph.compile()


full_weather_agent = build_full_weather_agent()


def ask_full_weather_ai(
    question: str,
    values: Optional[List[float]] = None
):

    result = full_weather_agent.invoke(
        {
            "question": question,
            "values": values,
            "route": "",
            "answer": "",
        }
    )

    return result


if __name__ == "__main__":

    print("=" * 60)
    print("FULL WEATHER INTELLIGENCE AI")
    print("=" * 60)

    question = input("\nEnter your question: ")

    values = None

    if "predict temperature" in question.lower():

        raw_values = input(
            "\nEnter 13 comma-separated weather values:\n"
        )

        values = [
            float(value.strip())
            for value in raw_values.split(",")
        ]

    result = ask_full_weather_ai(
        question,
        values
    )

    print("\nRoute Selected:", result["route"])
    print("\nFinal Answer:")
    print(result["answer"])