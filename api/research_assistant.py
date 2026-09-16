from fastapi import FastAPI
from pydantic import BaseModel

from api.agent.weather_agent import run_weather_agent


app = FastAPI(
    title="Weather Research Assistant API",
    description="RAG-based Weather Research Assistant",
    version="1.0"
)


class ResearchRequest(BaseModel):
    query: str


@app.get("/")
def root():
    return {
        "status": "success",
        "message": "Weather Research Assistant API is running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


@app.post("/research")
def research(request: ResearchRequest):

    if not request.query.strip():
        return {
            "success": False,
            "error": "Query cannot be empty"
        }

    result = run_weather_agent(request.query)

    return {
        "success": True,
        "query": request.query,
        "result": result
    }