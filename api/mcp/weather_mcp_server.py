import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mcp.server.mcpserver import MCPServer

from api.agent.weather_tool import predict_temperature


mcp = MCPServer("Weather Intelligence MCP Server")


@mcp.tool()
def predict_weather_temperature(values: list[float]) -> dict:
    """
    Predict temperature using the project's Random Forest weather model.

    The input must contain exactly 13 weather parameters.
    """

    if len(values) != 13:
        return {
            "success": False,
            "error": f"Expected 13 values, but received {len(values)}"
        }

    try:
        return predict_temperature(values)

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


@mcp.tool()
def weather_project_information() -> str:
    """
    Return information about the Weather Intelligence AI project.
    """

    return (
        "Weather Intelligence AI uses RandomForestRegressor as the "
        "primary deployed machine learning model for temperature "
        "prediction. The project uses FastAPI, semantic FAISS, "
        "LangChain, LangGraph, and Llama 3.2 3B."
    )


if __name__ == "__main__":
    mcp.run()