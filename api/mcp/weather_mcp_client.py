import anyio
import json

from mcp import Client, StdioServerParameters


async def call_mcp_prediction(values):

    server = StdioServerParameters(
        command="python",
        args=["-m", "api.mcp.weather_mcp_server"],
    )

    async with Client(server) as client:

        result = await client.call_tool(
            "predict_weather_temperature",
            {"values": values},
        )

        if result.is_error:
            error_text = "Unknown MCP error."

            if result.content:
                error_text = "\n".join(
                    getattr(item, "text", str(item))
                    for item in result.content
                )

            return {
                "success": False,
                "error": "MCP prediction tool returned an error.",
                "details": error_text
            }

        if result.content:

            text = result.content[0].text

            try:
                return json.loads(text)

            except json.JSONDecodeError:

                return {
                    "success": False,
                    "error": "MCP returned invalid JSON.",
                    "raw_response": text
                }

        return {
            "success": False,
            "error": "MCP returned an empty result."
        }


async def call_mcp_project_information():

    server = StdioServerParameters(
        command="python",
        args=["-m", "api.mcp.weather_mcp_server"],
    )

    async with Client(server) as client:

        result = await client.call_tool(
            "weather_project_information",
            {},
        )

        if result.is_error:
            return {
                "success": False,
                "error": "MCP project information tool returned an error."
            }

        if result.content:

            text = result.content[0].text

            return {
                "result": text
            }

        return {
            "success": False,
            "error": "MCP returned an empty result."
        }


def mcp_predict_temperature(values):

    return anyio.run(
        call_mcp_prediction,
        values
    )


def mcp_project_information():

    return anyio.run(
        call_mcp_project_information
    )


if __name__ == "__main__":

    print("=" * 60)
    print("MCP CLIENT TEST")
    print("=" * 60)

    values = [
        20,
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

    print("\nTesting MCP project information...\n")

    info = mcp_project_information()

    print(info)

    print("\nTesting MCP prediction...\n")

    prediction = mcp_predict_temperature(values)

    print(prediction)