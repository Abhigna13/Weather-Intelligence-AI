from typing import TypedDict

from langgraph.graph import StateGraph, START, END

from api.rag.langchain_rag import langchain_research


class WeatherAgentState(TypedDict):
    question: str
    answer: str


def research_node(state: WeatherAgentState):
    question = state["question"]

    answer = langchain_research(question)

    return {
        "answer": answer
    }


def build_weather_graph():
    graph = StateGraph(WeatherAgentState)

    graph.add_node("research", research_node)

    graph.add_edge(START, "research")
    graph.add_edge("research", END)

    return graph.compile()


weather_graph = build_weather_graph()


def ask_langgraph_weather_ai(question: str):
    result = weather_graph.invoke({
        "question": question,
        "answer": ""
    })

    return result["answer"]


if __name__ == "__main__":
    question = input(
        "Ask Weather Intelligence AI: "
    ).strip()

    if not question:
        print("Please enter a question.")
    else:
        answer = ask_langgraph_weather_ai(question)

        print("\n" + "=" * 60)
        print("LANGGRAPH WEATHER INTELLIGENCE AI")
        print("=" * 60)

        print("\nUser Question:")
        print(question)

        print("\nLangGraph Answer:")
        print("-" * 60)
        print(answer)
        print("-" * 60)