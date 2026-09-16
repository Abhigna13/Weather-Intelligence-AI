from api.agent.analytical_satellite_agent import run_analytical_workflow
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from api.model_loader import load_model, load_features
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

import pandas as pd
import json
import os
from api.agent.full_weather_agent import ask_full_weather_ai
from datetime import datetime
from api.agent.weather_agent import run_weather_agent
from api.rag.langchain_rag import langchain_research
from api.agent.langraph_weather_agent import ask_langgraph_weather_ai


# ============================================================
# WEATHER INTELLIGENCE AI
# FASTAPI BACKEND
# ============================================================


# ============================================================
# APPLICATION CONFIGURATION
# ============================================================

app = FastAPI(
    title="Weather Intelligence AI",
    description="AI-powered weather temperature prediction API",
    version="1.0.0"
)


# ============================================================
# FILE CONFIGURATION
# ============================================================

FRONTEND_DIRECTORY = "api/frontend"

FRONTEND_FILE = "api/frontend/index.html"

ANALYTICS_FILE = "api/frontend/analytics.html"

MODEL_PERFORMANCE_FILE = (
    "api/frontend/model-performance.html"
)

PREDICTION_HISTORY_FILE = (
    "api/frontend/prediction-history.html"
)

HISTORY_FILE = "prediction_history.json"

EVALUATION_FILE = "reports/results/evaluation_summary.json"


# ============================================================
# CORS CONFIGURATION
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# FRONTEND STATIC FILES
# ============================================================

app.mount(
    "/frontend",
    StaticFiles(
        directory=FRONTEND_DIRECTORY
    ),
    name="frontend"
)


# ============================================================
# PREDICTION HISTORY FUNCTIONS
# ============================================================

def load_prediction_history():

    if not os.path.exists(HISTORY_FILE):
        return []

    try:

        with open(
            HISTORY_FILE,
            "r",
            encoding="utf-8"
        ) as file:

            data = json.load(file)

            if isinstance(data, list):
                return data

            return []

    except (
        json.JSONDecodeError,
        OSError
    ):

        return []


def save_prediction_history(history):

    try:

        with open(
            HISTORY_FILE,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                history,
                file,
                indent=4,
                ensure_ascii=False
            )

    except OSError as e:

        raise HTTPException(
            status_code=500,
            detail={
                "message":
                    "Unable to save prediction history.",
                "error":
                    str(e)
            }
        )


def add_prediction_to_history(
    prediction,
    input_data
):

    history = load_prediction_history()

    existing_ids = [
        item.get("id", 0)
        for item in history
        if isinstance(item, dict)
    ]

    next_id = (
        max(existing_ids, default=0) + 1
    )

    record = {

        "id":
            next_id,

        "timestamp":
            datetime.now().strftime(
                "%d %b %Y • %I:%M %p"
            ),

        "predicted_temperature":
            round(
                float(prediction),
                2
            ),

        "summary":
            input_data.get("summary"),

        "precip_type":
            input_data.get("precip_type"),

        "apparent_temperature":
            input_data.get(
                "apparent_temperature"
            ),

        "humidity":
            input_data.get("humidity"),

        "wind_speed":
            input_data.get("wind_speed"),

        "wind_bearing":
            input_data.get(
                "wind_bearing"
            ),

        "visibility":
            input_data.get("visibility"),

        "cloud_cover":
            input_data.get(
                "cloud_cover"
            ),

        "pressure":
            input_data.get("pressure"),

        "daily_summary":
            input_data.get(
                "daily_summary"
            ),

        "year":
            input_data.get("year"),

        "month":
            input_data.get("month"),

        "day":
            input_data.get("day")
    }

    # Newest prediction first
    history.insert(
        0,
        record
    )

    save_prediction_history(
        history
    )

    return record


# ============================================================
# FRONTEND PAGES
# ============================================================

@app.get("/app")
def frontend():

    return FileResponse(
        FRONTEND_FILE
    )


@app.get("/analytics")
def analytics_page():

    return FileResponse(
        ANALYTICS_FILE
    )


@app.get("/model-performance-page")
def model_performance_page():

    return FileResponse(
        MODEL_PERFORMANCE_FILE
    )


@app.get("/prediction-history-page")
def prediction_history_page():

    return FileResponse(
        PREDICTION_HISTORY_FILE
    )


# ============================================================
# PREDICTION HISTORY API
# ============================================================

@app.get("/prediction-history")
def get_prediction_history():

    history = load_prediction_history()

    return {

        "success":
            True,

        "count":
            len(history),

        "history":
            history
    }


# ============================================================
# LOAD AI MODEL AND FEATURES
# ============================================================

try:

    model = load_model()

    features = load_features()

    MODEL_STATUS = True

    MODEL_ERROR = None

except Exception as e:

    model = None

    features = []

    MODEL_STATUS = False

    MODEL_ERROR = str(e)


# ============================================================
# INPUT DATA SCHEMA
# ============================================================

class WeatherInput(BaseModel):

    summary: float = Field(
        ...,
        description="Encoded weather summary"
    )

    precip_type: float = Field(
        ...,
        description="Encoded precipitation type"
    )

    apparent_temperature: float = Field(
        ...,
        description="Apparent temperature in Celsius"
    )

    humidity: float = Field(
        ...,
        ge=0,
        le=1,
        description="Relative humidity between 0 and 1"
    )

    wind_speed: float = Field(
        ...,
        ge=0,
        description="Wind speed in km/h"
    )

    wind_bearing: float = Field(
        ...,
        ge=0,
        le=360,
        description="Wind bearing in degrees"
    )

    visibility: float = Field(
        ...,
        ge=0,
        description="Visibility in km"
    )

    cloud_cover: float = Field(
        ...,
        description="Cloud cover value"
    )

    pressure: float = Field(
        ...,
        ge=0,
        description="Atmospheric pressure in millibars"
    )

    daily_summary: float = Field(
        ...,
        description="Encoded daily weather summary"
    )

    year: int = Field(
        ...,
        ge=1900,
        le=2100,
        description="Year"
    )

    month: int = Field(
        ...,
        ge=1,
        le=12,
        description="Month"
    )

    day: int = Field(
        ...,
        ge=1,
        le=31,
        description="Day"
    )

class ResearchRequest(BaseModel):

    query: str = Field(
        ...,
        min_length=1,
        description="Weather research question"
    )

# ============================================================
# ROOT ENDPOINT
# ============================================================

@app.get("/")
def home():

    return {

        "message":
            "Weather Intelligence AI API",

        "status":
            "running",

        "version":
            "1.0.0",

        "model_loaded":
            MODEL_STATUS,

        "documentation":
            "/docs"
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health_check():

    return {

        "status":
            "healthy"
            if MODEL_STATUS
            else "degraded",

        "service":
            "Weather Intelligence AI",

        "model_loaded":
            MODEL_STATUS
    }


# ============================================================
# MODEL INFORMATION
# ============================================================

@app.get("/model-info")
def model_info():

    if not MODEL_STATUS:

        return {

            "model_loaded":
                False,

            "features":
                [],

            "feature_count":
                0,

            "error":
                MODEL_ERROR
        }

    return {

        "model_loaded":
            True,

        "model_name":
            type(model).__name__,

        "feature_count":
            len(features),

        "features":
            features
    }


# ============================================================
# MODEL PERFORMANCE
# ============================================================

@app.get("/model-performance")
def model_performance():

    if not os.path.exists(
        EVALUATION_FILE
    ):

        raise HTTPException(
            status_code=404,
            detail="Evaluation summary file not found."
        )

    try:

        with open(
            EVALUATION_FILE,
            "r",
            encoding="utf-8"
        ) as file:

            evaluation = json.load(file)

        return {

            "success":
                True,

            "evaluation":
                evaluation
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail={
                "message":
                    "Unable to load evaluation data.",
                "error":
                    str(e)
            }
        )


# ============================================================
# TEMPERATURE PREDICTION
# ============================================================

@app.post("/predict")
def predict(data: WeatherInput):

    # --------------------------------------------------------
    # CHECK MODEL
    # --------------------------------------------------------

    if (
        not MODEL_STATUS
        or model is None
    ):

        raise HTTPException(
            status_code=503,
            detail=(
                "AI model is not available. "
                "Please check the model files."
            )
        )

    # --------------------------------------------------------
    # PREPARE INPUT DATA
    # --------------------------------------------------------

    input_data = {

        "Summary":
            data.summary,

        "Precip Type":
            data.precip_type,

        "Apparent Temperature (C)":
            data.apparent_temperature,

        "Humidity":
            data.humidity,

        "Wind Speed (km/h)":
            data.wind_speed,

        "Wind Bearing (degrees)":
            data.wind_bearing,

        "Visibility (km)":
            data.visibility,

        "Loud Cover":
            data.cloud_cover,

        "Pressure (millibars)":
            data.pressure,

        "Daily Summary":
            data.daily_summary,

        "Year":
            data.year,

        "Month":
            data.month,

        "Day":
            data.day
    }

    try:

        # ----------------------------------------------------
        # CREATE DATAFRAME
        # ----------------------------------------------------

        input_df = pd.DataFrame(
            [input_data]
        )

        # ----------------------------------------------------
        # VALIDATE MODEL FEATURES
        # ----------------------------------------------------

        if not features:

            raise HTTPException(
                status_code=500,
                detail="Model feature configuration is empty."
            )

        missing_features = [

            feature

            for feature in features

            if feature not in input_df.columns
        ]

        if missing_features:

            raise HTTPException(
                status_code=500,
                detail={
                    "message":
                        "Required model features are missing.",
                    "missing_features":
                        missing_features
                }
            )

        # ----------------------------------------------------
        # MATCH EXACT FEATURE ORDER
        # ----------------------------------------------------

        input_df = input_df[
            features
        ]

        # ----------------------------------------------------
        # AI PREDICTION
        # ----------------------------------------------------

        prediction = model.predict(
            input_df
        )

        predicted_temperature = round(
            float(prediction[0]),
            2
        )

        # ----------------------------------------------------
        # SAVE PREDICTION HISTORY
        # ----------------------------------------------------

        history_input = {

            "summary":
                data.summary,

            "precip_type":
                data.precip_type,

            "apparent_temperature":
                data.apparent_temperature,

            "humidity":
                data.humidity,

            "wind_speed":
                data.wind_speed,

            "wind_bearing":
                data.wind_bearing,

            "visibility":
                data.visibility,

            "cloud_cover":
                data.cloud_cover,

            "pressure":
                data.pressure,

            "daily_summary":
                data.daily_summary,

            "year":
                data.year,

            "month":
                data.month,

            "day":
                data.day
        }

        history_record = add_prediction_to_history(
            predicted_temperature,
            history_input
        )

        # ----------------------------------------------------
        # RETURN PREDICTION
        # ----------------------------------------------------

        return {

            "success":
                True,

            "prediction": {

                "predicted_temperature_celsius":
                    predicted_temperature
            },

            "model": {

                "model_loaded":
                    True,

                "model_name":
                    type(model).__name__,

                "features_used":
                    len(features)
            },

            "history": {

                "saved":
                    True,

                "record_id":
                    history_record["id"]
            }
        }

    except HTTPException:

        raise

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail={
                "message":
                    "Prediction failed.",
                "error":
                    str(e)
            }
        )


# ============================================================
# DYNAMIC ANALYTICS
# ============================================================

@app.get("/analytics-data")
def analytics_data():

    history = load_prediction_history()

    if not history:

        return {

            "total_predictions":
                0,

            "latest_prediction":
                None,

            "average_prediction":
                None,

            "highest_prediction":
                None,

            "lowest_prediction":
                None,

            "history":
                []
        }

    temperatures = [

        float(
            item["predicted_temperature"]
        )

        for item in history

        if (
            isinstance(item, dict)
            and "predicted_temperature" in item
        )
    ]

    if not temperatures:

        return {

            "total_predictions":
                0,

            "latest_prediction":
                None,

            "average_prediction":
                None,

            "highest_prediction":
                None,

            "lowest_prediction":
                None,

            "history":
                history
        }

    total_predictions = len(
        temperatures
    )

    latest_prediction = temperatures[0]

    average_prediction = round(
        sum(temperatures)
        /
        total_predictions,
        2
    )

    highest_prediction = max(
        temperatures
    )

    lowest_prediction = min(
        temperatures
    )

    return {

        "total_predictions":
            total_predictions,

        "latest_prediction":
            latest_prediction,

        "average_prediction":
            average_prediction,

        "highest_prediction":
            highest_prediction,

        "lowest_prediction":
            lowest_prediction,

        "history":
            history
    }


# ============================================================
# CLEAR PREDICTION HISTORY
# ============================================================

@app.delete("/prediction-history")
def clear_prediction_history():

    save_prediction_history([])

    return {

        "success":
            True,

        "message":
            "Prediction history cleared.",

        "count":
            0
    }


# ============================================================
# APPLICATION INFORMATION
# ============================================================

@app.get("/api-info")
def api_info():

    return {

        "name":
            "Weather Intelligence AI",

        "description":
            "AI-powered weather temperature prediction system",

        "version":
            "1.0.0",

        "status":
            "online",

        "model_loaded":
            MODEL_STATUS,

        "endpoints": {

            "home":
                "/",

            "health":
                "/health",

            "model_info":
                "/model-info",

            "model_performance":
                "/model-performance",

            "prediction":
                "/predict",

            "prediction_history":
                "/prediction-history",

            "analytics":
                "/analytics-data",

            "clear_history":
                "/prediction-history",

            "frontend":
                "/app",

            "analytics_page":
                "/analytics",

            "model_performance_page":
                "/model-performance-page",

            "prediction_history_page":
                "/prediction-history-page",

            "documentation":
                "/docs"
        }
    }

# ============================================================
# WEATHER RESEARCH ASSISTANT
# ============================================================

@app.post("/research")
def research_assistant(data: ResearchRequest):

    try:

        query = data.query.strip()

        if not query:
            raise HTTPException(
                status_code=400,
                detail="Research query cannot be empty."
            )

        result = run_weather_agent(query)

        return {
            "success": True,
            "query": query,
            "result": result
        }

    except HTTPException:
        raise

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail={
                "message":
                    "Weather Research Assistant failed.",
                "error":
                    str(e)
            }
        )


# ============================================================
# LANGCHAIN WEATHER RESEARCH ASSISTANT
# ============================================================

@app.post("/research-langchain")
def research_langchain(data: ResearchRequest):

    try:

        query = data.query.strip()

        if not query:
            raise HTTPException(
                status_code=400,
                detail="Research query cannot be empty."
            )

        result = langchain_research(query)

        return {
            "success": True,
            "query": query,
            "result": result,
            "engine":
                "LangChain + Semantic FAISS + Llama 3.2 3B"
        }

    except HTTPException:
        raise

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail={
                "message":
                    "LangChain Research Assistant failed.",
                "error":
                    str(e)
            }
        )


# ============================================================
# LANGGRAPH WEATHER RESEARCH ASSISTANT
# ============================================================

@app.post("/research-langgraph")
def research_langgraph(data: ResearchRequest):

    try:

        query = data.query.strip()

        if not query:
            raise HTTPException(
                status_code=400,
                detail="Research query cannot be empty."
            )

        result = ask_langgraph_weather_ai(query)

        return {
            "success": True,
            "query": query,
            "result": result,
            "engine":
                "LangGraph + LangChain + Semantic FAISS + Llama 3.2 3B"
        }

    except HTTPException:
        raise

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail={
                "message":
                    "LangGraph Research Assistant failed.",
                "error":
                    str(e)
            }
        )

class FullWeatherAIRequest(BaseModel):
    question: str
    values: list[float] | None = None

class AnalyticalWeatherRequest(BaseModel):
    question: str
    values: list[float]


@app.post("/research-full")
def research_full(request: FullWeatherAIRequest):

    try:

        result = ask_full_weather_ai(
            request.question,
            request.values
        )

        return {
            "success": True,
            "query": request.question,
            "result": result
        }

    except Exception as e:

        return {
            "success": False,
            "error": str(e)
        }

# ============================================================
# INTEGRATED WEATHER + SATELLITE ANALYTICAL WORKFLOW
# ============================================================

@app.post("/analytical-weather")
def analytical_weather(request: AnalyticalWeatherRequest):

    try:

        if not request.question.strip():
            raise HTTPException(
                status_code=400,
                detail="Analytical question cannot be empty."
            )

        if len(request.values) != 13:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Exactly 13 weather parameters "
                    "are required."
                )
            )

        result = run_analytical_workflow(
            question=request.question,
            values=request.values
        )

        return {
            "success": True,
            "query": request.question,
            "result": result,
            "workflow": {
                "scientific_knowledge": True,
                "mcp_weather_prediction": True,
                "satellite_analysis": True,
                "integrated_interpretation": True
            }
        }

    except HTTPException:
        raise

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail={
                "message":
                    "Integrated analytical workflow failed.",
                "error":
                    str(e)
            }
        )

@app.get("/research-assistant")
def research_assistant_page():
    return FileResponse(
        os.path.join(
            FRONTEND_DIRECTORY,
            "research-assistant.html"
        )
    )