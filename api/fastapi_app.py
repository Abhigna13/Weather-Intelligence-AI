from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import numpy as np

from api.model_loader import load_model, load_features

app = FastAPI(
    title="Weather Intelligence AI API",
    description="FastAPI backend for weather temperature prediction",
    version="1.0.0"
)

model = load_model()
feature_columns = load_features()


class WeatherInput(BaseModel):
    values: List[float]


@app.get("/")
def root():
    return {
        "message": "Weather Intelligence AI API is running",
        "model": "Weather AI Model",
        "features": len(feature_columns)
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "feature_count": len(feature_columns)
    }


@app.post("/predict")
def predict(data: WeatherInput):

    if len(data.values) != len(feature_columns):
        return {
            "error": f"Expected {len(feature_columns)} values, "
                     f"but received {len(data.values)}"
        }

    input_data = np.array(data.values).reshape(1, -1)

    prediction = model.predict(input_data)

    return {
        "prediction": float(prediction[0]),
        "unit": "°C",
        "model": "Weather AI Model"
    }