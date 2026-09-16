# ============================================================
# AGENT-DRIVEN ANALYTICAL WORKFLOW
# WEATHER + SCIENTIFIC KNOWLEDGE + SATELLITE INTEGRATION
# ============================================================

import sys
import json
from pathlib import Path
from typing import TypedDict, Optional, List, Any


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# IMPORTS
# ============================================================

from langgraph.graph import StateGraph, START, END

from api.rag.langchain_rag import langchain_research
from api.mcp.weather_mcp_client import mcp_predict_temperature


# ============================================================
# PATHS
# ============================================================

SATELLITE_RESULT_FILE = (
    PROJECT_ROOT
    / "reports"
    / "satellite"
    / "remote_sensing_integration.json"
)

OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "reports"
    / "analytical_workflow"
)

OUTPUT_FILE = (
    OUTPUT_DIRECTORY
    / "integrated_weather_satellite_analysis.json"
)


# ============================================================
# STATE
# ============================================================

class AnalyticalWeatherState(TypedDict):

    question: str

    values: Optional[List[float]]

    scientific_knowledge: str

    weather_prediction: Any

    predicted_temperature: Optional[float]

    satellite_analysis: Any

    satellite_classification: str

    integrated_analysis: str


# ============================================================
# NODE 1
# SCIENTIFIC KNOWLEDGE RETRIEVAL
# ============================================================

def scientific_research_node(
    state: AnalyticalWeatherState
):

    print(
        "\n[1/4] Running scientific knowledge retrieval..."
    )

    try:

        answer = langchain_research(
            state["question"]
        )

        print(
            "✓ Scientific knowledge retrieval completed."
        )

    except Exception as exc:

        answer = (
            "Scientific knowledge retrieval failed: "
            f"{exc}"
        )

        print(
            f"⚠ Scientific retrieval failed: {exc}"
        )

    return {

        "scientific_knowledge":
            answer

    }


# ============================================================
# NODE 2
# DIRECT MCP WEATHER PREDICTION
# ============================================================

def weather_prediction_node(
    state: AnalyticalWeatherState
):

    print(
        "\n[2/4] Running MCP weather prediction..."
    )

    values = state.get("values")


    if values is None:

        return {

            "weather_prediction": {

                "success": False,

                "error":
                    "13 weather parameters are required."

            },

            "predicted_temperature":
                None

        }


    # --------------------------------------------------------
    # Validate number of parameters
    # --------------------------------------------------------

    if len(values) != 13:

        return {

            "weather_prediction": {

                "success": False,

                "error":
                    f"Expected 13 values, but received {len(values)}"

            },

            "predicted_temperature":
                None

        }


    # --------------------------------------------------------
    # Call MCP prediction client
    # --------------------------------------------------------

    try:

        prediction_result = (
            mcp_predict_temperature(
                values
            )
        )

    except Exception as exc:

        print(
            f"⚠ MCP prediction failed: {exc}"
        )

        return {

            "weather_prediction": {

                "success": False,

                "error":
                    str(exc)

            },

            "predicted_temperature":
                None

        }


    # --------------------------------------------------------
    # Validate MCP response type
    # --------------------------------------------------------

    if not isinstance(
        prediction_result,
        dict
    ):

        print(
            "⚠ MCP returned an unexpected response."
        )

        return {

            "weather_prediction":
                prediction_result,

            "predicted_temperature":
                None

        }


    # --------------------------------------------------------
    # Check MCP success
    # --------------------------------------------------------

    if not prediction_result.get(
        "success",
        False
    ):

        print(
            "⚠ MCP prediction returned an error."
        )

        return {

            "weather_prediction":
                prediction_result,

            "predicted_temperature":
                None

        }


    # --------------------------------------------------------
    # Extract prediction object
    # --------------------------------------------------------

    prediction = (
        prediction_result.get(
            "prediction"
        )
    )


    if not isinstance(
        prediction,
        dict
    ):

        print(
            "⚠ MCP response does not contain "
            "a prediction object."
        )

        return {

            "weather_prediction":
                prediction_result,

            "predicted_temperature":
                None

        }


    # --------------------------------------------------------
    # Extract predicted temperature
    # --------------------------------------------------------

    predicted_temperature = (
        prediction.get(
            "predicted_temperature_celsius"
        )
    )


    # --------------------------------------------------------
    # Convert to float
    # --------------------------------------------------------

    try:

        predicted_temperature = float(
            predicted_temperature
        )

    except (
        TypeError,
        ValueError
    ):

        predicted_temperature = None


    # --------------------------------------------------------
    # Display result
    # --------------------------------------------------------

    if predicted_temperature is not None:

        print(
            "✓ MCP weather prediction completed."
        )

        print(
            f"✓ Predicted temperature: "
            f"{predicted_temperature:.2f} °C"
        )

    else:

        print(
            "⚠ MCP prediction completed, "
            "but temperature is unavailable."
        )


    # --------------------------------------------------------
    # Return BOTH prediction and temperature
    # --------------------------------------------------------

    return {

        "weather_prediction":
            prediction_result,

        "predicted_temperature":
            predicted_temperature

    }


# ============================================================
# NODE 3
# SATELLITE CLOUD ANALYSIS
# ============================================================

def satellite_analysis_node(
    state: AnalyticalWeatherState
):

    print(
        "\n[3/4] Loading satellite cloud analysis..."
    )


    # --------------------------------------------------------
    # Check satellite result file
    # --------------------------------------------------------

    if not SATELLITE_RESULT_FILE.exists():

        print(
            "⚠ Satellite analysis result file not found."
        )

        return {

            "satellite_analysis": {

                "success": False,

                "message":
                    "Satellite analysis result file "
                    "not found."

            },

            "satellite_classification":
                "unknown"

        }


    # --------------------------------------------------------
    # Load JSON
    # --------------------------------------------------------

    try:

        with open(
            SATELLITE_RESULT_FILE,
            "r",
            encoding="utf-8"
        ) as file:

            data = json.load(file)

    except Exception as exc:

        print(
            f"⚠ Could not load satellite result: {exc}"
        )

        return {

            "satellite_analysis": {

                "success":
                    False,

                "message":
                    str(exc)

            },

            "satellite_classification":
                "unknown"

        }


    # --------------------------------------------------------
    # Recursive helper
    # --------------------------------------------------------

    def find_satellite_value(
        obj,
        keys
    ):

        if isinstance(
            obj,
            dict
        ):

            for key, value in obj.items():

                if str(key).lower() in keys:

                    if value is not None:

                        return value


            for value in obj.values():

                result = find_satellite_value(
                    value,
                    keys
                )

                if result is not None:

                    return result


        elif isinstance(
            obj,
            list
        ):

            for item in obj:

                result = find_satellite_value(
                    item,
                    keys
                )

                if result is not None:

                    return result


        return None


    # --------------------------------------------------------
    # Classification
    # --------------------------------------------------------

    classification = (
        find_satellite_value(
            data,
            {
                "classification",
                "cloud_classification",
                "predicted_class",
                "label"
            }
        )
    )


    # --------------------------------------------------------
    # Confidence
    # --------------------------------------------------------

    confidence = (
        find_satellite_value(
            data,
            {
                "confidence",
                "prediction_confidence",
                "confidence_score"
            }
        )
    )


    if classification is None:

        classification = "unknown"


    classification = str(
        classification
    )


    # --------------------------------------------------------
    # Satellite result
    # --------------------------------------------------------

    satellite_result = {

        "success":
            True,

        "classification":
            classification,

        "confidence":
            confidence,

        "source":
            "remote_sensing_integration.json"

    }


    print(
        "✓ Satellite analysis result loaded."
    )

    print(
        f"✓ Satellite classification: "
        f"{classification}"
    )


    if confidence is not None:

        print(
            f"✓ Satellite confidence: "
            f"{confidence}"
        )


    return {

        "satellite_analysis":
            satellite_result,

        "satellite_classification":
            classification

    }


# ============================================================
# NODE 4
# INTEGRATED ANALYSIS
# ============================================================

def integrated_analysis_node(
    state: AnalyticalWeatherState
):

    print(
        "\n[4/4] Generating integrated "
        "weather + satellite analysis..."
    )


    question = state[
        "question"
    ]


    scientific_knowledge = (
        state.get(
            "scientific_knowledge",
            ""
        )
    )


    predicted_temperature = (
        state.get(
            "predicted_temperature"
        )
    )


    satellite_classification = (
        state.get(
            "satellite_classification",
            "unknown"
        )
    )


    satellite_analysis = (
        state.get(
            "satellite_analysis",
            {}
        )
    )


    weather_prediction = (
        state.get(
            "weather_prediction",
            {}
        )
    )


    # --------------------------------------------------------
    # Temperature text
    # --------------------------------------------------------

    if predicted_temperature is not None:

        temperature_text = (
            f"{predicted_temperature:.2f} °C"
        )

    else:

        temperature_text = (
            "not available"
        )


    # ========================================================
    # INTEGRATED INTERPRETATION
    # ========================================================

    integrated_text = (

        "Integrated Weather Intelligence Analysis\n\n"

        f"Question:\n"
        f"{question}\n\n"

        "Weather Prediction:\n"
        f"The MCP-connected Random Forest weather "
        f"prediction workflow produced a predicted "
        f"temperature of {temperature_text}.\n\n"

        "Satellite Cloud Analysis:\n"
        f"The remote-sensing workflow classified the "
        f"observed satellite cloud condition as "
        f"{satellite_classification}.\n\n"

        "Integrated Interpretation:\n"
        f"The weather prediction and satellite cloud "
        f"classification provide complementary sources "
        f"of weather intelligence. The machine learning "
        f"workflow provides the numerical temperature "
        f"estimate, while the satellite workflow provides "
        f"cloud-condition information from remote sensing. "
        f"Together, these outputs provide a combined view "
        f"of the analyzed atmospheric conditions.\n\n"

        "Scientific Knowledge Context:\n"
        f"The retrieved scientific knowledge is presented "
        f"separately above and is used as the research "
        f"context for interpreting the weather and satellite "
        f"outputs.\n\n"

        "Limitation:\n"
        f"The satellite classification is a proof-of-concept "
        f"result based on the available satellite dataset "
        f"and implemented remote-sensing workflow."
    )


    # --------------------------------------------------------
    # Final JSON
    # --------------------------------------------------------

    integrated_result = {

        "success":
            True,

        "question":
            question,

        "weather_prediction":
            weather_prediction,

        "predicted_temperature_celsius":
            predicted_temperature,

        "satellite_analysis":
            satellite_analysis,

        "satellite_classification":
            satellite_classification,

        "scientific_knowledge":
            scientific_knowledge,

        "integrated_interpretation":
            integrated_text

    }


    # --------------------------------------------------------
    # Save result
    # --------------------------------------------------------

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True
    )


    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            integrated_result,
            file,
            indent=4,
            ensure_ascii=False,
            default=str
        )


    print(
        "✓ Integrated analysis saved."
    )

    print(
        f"✓ Output file: {OUTPUT_FILE}"
    )


    # ========================================================
    # CRITICAL FIX
    #
    # Return the actual prediction temperature and other
    # analytical fields through the LangGraph state.
    # ========================================================

    return {

        "integrated_analysis":
            integrated_text,

        "weather_prediction":
            weather_prediction,

        "predicted_temperature":
            predicted_temperature,

        "satellite_analysis":
            satellite_analysis,

        "satellite_classification":
            satellite_classification,

        "scientific_knowledge":
            scientific_knowledge

    }


# ============================================================
# BUILD LANGGRAPH WORKFLOW
# ============================================================

def build_analytical_workflow():

    graph = StateGraph(
        AnalyticalWeatherState
    )


    graph.add_node(
        "scientific_research",
        scientific_research_node
    )

    graph.add_node(
        "weather_prediction",
        weather_prediction_node
    )

    graph.add_node(
        "satellite_analysis",
        satellite_analysis_node
    )

    graph.add_node(
        "integrated_analysis",
        integrated_analysis_node
    )


    # --------------------------------------------------------
    # Sequential workflow
    # --------------------------------------------------------

    graph.add_edge(
        START,
        "scientific_research"
    )

    graph.add_edge(
        "scientific_research",
        "weather_prediction"
    )

    graph.add_edge(
        "weather_prediction",
        "satellite_analysis"
    )

    graph.add_edge(
        "satellite_analysis",
        "integrated_analysis"
    )

    graph.add_edge(
        "integrated_analysis",
        END
    )


    return graph.compile()


# ============================================================
# RUN WORKFLOW
# ============================================================

def run_analytical_workflow(
    question=None,
    values=None
):

    print(
        "=" * 70
    )

    print(
        "AGENT-DRIVEN ANALYTICAL WORKFLOW"
    )

    print(
        "WEATHER + SCIENTIFIC KNOWLEDGE + "
        "SATELLITE INTEGRATION"
    )

    print(
        "=" * 70
    )


    # --------------------------------------------------------
    # Default question
    # --------------------------------------------------------

    if question is None:

        question = (
            "Analyze the weather prediction using scientific "
            "knowledge and satellite cloud information."
        )


    # --------------------------------------------------------
    # Default weather values
    # --------------------------------------------------------

    if values is None:

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


    print(
        "\nQuestion:"
    )

    print(
        question
    )


    print(
        "\nWeather input:"
    )

    print(
        values
    )


    print(
        "\nBuilding LangGraph workflow..."
    )


    workflow = (
        build_analytical_workflow()
    )


    print(
        "LangGraph workflow created."
    )


    print(
        "\nExecuting integrated workflow..."
    )


    # --------------------------------------------------------
    # Initial state
    # --------------------------------------------------------

    initial_state: AnalyticalWeatherState = {

        "question":
            question,

        "values":
            values,

        "scientific_knowledge":
            "",

        "weather_prediction":
            {},

        "predicted_temperature":
            None,

        "satellite_analysis":
            {},

        "satellite_classification":
            "unknown",

        "integrated_analysis":
            ""

    }


    # --------------------------------------------------------
    # Execute workflow
    # --------------------------------------------------------

    result = workflow.invoke(
        initial_state
    )


    # --------------------------------------------------------
    # Display result
    # --------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "INTEGRATED ANALYTICAL RESULT"
    )

    print(
        "=" * 70
    )

    print(
        result[
            "integrated_analysis"
        ]
    )


    print(
        "\nPredicted temperature returned by workflow:"
    )

    print(
        result.get(
            "predicted_temperature"
        )
    )


    print(
        "\nSatellite classification returned by workflow:"
    )

    print(
        result.get(
            "satellite_classification"
        )
    )


    print(
        "\n" + "=" * 70
    )

    print(
        "AGENT-DRIVEN ANALYTICAL WORKFLOW "
        "COMPLETED SUCCESSFULLY"
    )

    print(
        "=" * 70
    )


    print(
        "\nSaved:"
    )

    print(
        OUTPUT_FILE
    )


    return result


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    run_analytical_workflow()