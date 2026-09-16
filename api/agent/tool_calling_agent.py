from typing import TypedDict, Optional, List

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver

from api.agent.weather_tool import predict_temperature
from api.rag.langchain_rag import langchain_research


class WeatherToolState(TypedDict):
    question: str
    tool_used: str
    values: Optional[List]
    answer: str
    conversation_history: List[str]


def select_tool(state: WeatherToolState):
    question = state["question"].lower()

    prediction_keywords = [
        "predict temperature",
        "predict weather",
        "make a prediction",
        "give me a prediction",
        "forecast temperature"
    ]

    if any(keyword in question for keyword in prediction_keywords):
        return {"tool_used": "prediction"}

    return {"tool_used": "research"}


def prediction_tool_node(state: WeatherToolState):
    values = state.get("values")

    if not values:
        answer = "Please provide the 13 weather parameters for temperature prediction."
    else:
        result = predict_temperature(values)
        answer = str(result)

    history = state.get("conversation_history", [])

    history.append(
        f"User: {state['question']}\nAssistant: {answer}"
    )

    return {
        "answer": answer,
        "conversation_history": history
    }


def research_tool_node(state: WeatherToolState):
    question = state["question"]

    history = state.get("conversation_history", [])

    previous_context = ""

    if history:
        previous_context = (
            "\nPrevious conversation:\n"
            + "\n".join(history[-4:])
            + "\n"
        )

    enhanced_question = question

    if previous_context:
        enhanced_question = (
            previous_context
            + "\nCurrent user question:\n"
            + question
        )

    answer = langchain_research(enhanced_question)

    history.append(
        f"User: {question}\nAssistant: {answer}"
    )

    return {
        "answer": answer,
        "conversation_history": history
    }


def route_tool(state: WeatherToolState):
    if state["tool_used"] == "prediction":
        return "prediction"

    return "research"


def build_weather_tool_agent():

    graph = StateGraph(WeatherToolState)

    graph.add_node("select_tool", select_tool)
    graph.add_node("prediction", prediction_tool_node)
    graph.add_node("research", research_tool_node)

    graph.add_edge(START, "select_tool")

    graph.add_conditional_edges(
        "select_tool",
        route_tool,
        {
            "prediction": "prediction",
            "research": "research"
        }
    )

    graph.add_edge("prediction", END)
    graph.add_edge("research", END)

    memory = InMemorySaver()

    return graph.compile(checkpointer=memory)


weather_tool_agent = build_weather_tool_agent()


def ask_weather_tool_agent(
    question: str,
    values=None,
    thread_id="weather-session-1"
):

    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    result = weather_tool_agent.invoke(
        {
            "question": question,
            "tool_used": "",
            "values": values,
            "answer": ""
        },
        config=config
    )

    return result


if __name__ == "__main__":

    print("=" * 70)
    print("WEATHER AI TOOL-CALLING AGENT")
    print("=" * 70)

    question = input("\nEnter your question: ")

    values = None

    if "predict temperature" in question.lower():

        print("\nEnter 13 weather parameters separated by commas:")

        raw_values = input()

        values = [
            float(value.strip())
            for value in raw_values.split(",")
        ]

    result = ask_weather_tool_agent(
        question,
        values=values
    )

    print("\nTool Selected:")
    print(result["tool_used"])

    print("\nAgent Answer:")
    print(result["answer"])

    print("\nConversation History:")
    for item in result["conversation_history"]:
        print(item)

    print("\n" + "=" * 70)