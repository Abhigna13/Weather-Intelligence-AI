# Installation Guide

## 1. Project Requirements

The Weather Intelligence and Climate Decision Support Platform is developed using Python and related AI/ML libraries.

### Required Software

- Python 3.9 or later
- Visual Studio Code
- Git
- Web browser
- Docker (optional; required only for container-based deployment)
- Ollama (required for the local Llama 3.2 model)

---

## 2. Project Location

The project should be placed in a local development directory.

Example:

    C:\Users\HP\Documents\Weather-Intelligence-AI

---

## 3. Open the Project in VS Code

If the `code .` command is available, run:

    cd "C:\Users\HP\Documents\Weather-Intelligence-AI"
    code .

Alternatively, open Visual Studio Code and select the project folder manually.

---

## 4. Create a Virtual Environment

Open the VS Code terminal and run:

    python -m venv venv

This creates an isolated Python virtual environment for the project.

---

## 5. Activate the Virtual Environment

For Windows PowerShell, run:

    .\venv\Scripts\Activate.ps1

After activation, `(venv)` should appear at the beginning of the terminal prompt.

---

## 6. Install Required Libraries

Run the following commands:

    python -m pip install --upgrade pip
    pip install -r requirements.txt

The required Python libraries for data processing, machine learning, deep learning, FastAPI, RAG, LangChain, LangGraph, SHAP, and related components will be installed.

---

## 7. Verify Python Installation

Run:

    python --version

The project uses Python 3.9 or later.

---

## 8. Verify the Raw Dataset

The raw weather dataset should be located at:

    datasets/raw/weather.csv

The dataset contains weather-related variables such as:

- Temperature
- Apparent Temperature
- Humidity
- Wind Speed
- Wind Bearing
- Visibility
- Pressure
- Precipitation Type
- Date and time information

---

## 9. Run Data Preprocessing

Run:

    python api/preprocessing/preprocessing.py

The processed dataset will be generated at:

    datasets/processed/weather_preprocessed.csv

The preprocessing stage handles data cleaning, transformation, missing-value handling, and preparation of the dataset for further analysis.

---

## 10. Run Exploratory Data Analysis

Run:

    python api/eda/eda_analysis.py

The EDA process generates statistical summaries, distributions, correlations, and temperature-related analysis.

Generated outputs are stored under:

    reports/eda/

Important outputs include:

    reports/eda/descriptive_statistics.csv
    reports/eda/correlation_matrix.csv
    reports/eda/correlation_matrix.png
    reports/eda/temperature_distribution.png
    reports/eda/humidity_distribution.png
    reports/eda/pressure_distribution.png
    reports/eda/wind_speed_distribution.png
    reports/eda/monthly_temperature.csv
    reports/eda/monthly_temperature.png

---

## 11. Run Feature Engineering

Run:

    python api/feature_engineering/feature_engineering.py

The feature engineering stage creates additional weather-related features and transformations.

The engineered dataset will be generated at:

    datasets/processed/weather_features.csv

---

## 12. Prepare Machine Learning Data

Run:

    python api/models/prepare_data.py

Generated files:

    datasets/processed/X_train_ml.csv
    datasets/processed/X_test_ml.csv
    datasets/processed/y_train_ml.csv
    datasets/processed/y_test_ml.csv

The data is divided into training and testing datasets for machine learning model development and evaluation.

---

## 13. Train Machine Learning Models

Run:

    python api/models/train_models.py

The following machine learning models are trained and evaluated:

- Linear Regression
- Decision Tree
- Random Forest
- Gradient Boosting

The trained models are saved in the project model directory.

---

## 14. Model Comparison

Run:

    python api/models/model_comparison.py

The model comparison evaluates the implemented machine learning models using performance metrics such as:

- MAE
- MSE
- RMSE
- R²

The generated comparison results are stored under:

    reports/results/

The Random Forest model achieved the strongest performance among the implemented machine learning models in the current experiments.

---

## 15. Train Deep Learning Models

The project includes deep learning models for temperature prediction.

Run the LSTM model:

    python api/models/deep_learning/lstm_model.py

The trained LSTM model is saved at:

    api/models/deep_learning/lstm_temperature_model.keras

Additional deep learning models evaluated in the project include:

- LSTM
- GRU
- CNN-LSTM

The comparison results are generated in:

    reports/results/deep_learning_model_comparison.csv
    reports/results/deep_learning_model_comparison.png
    reports/results/deep_learning_rmse_comparison.png

---

## 16. Run Time-Series Forecasting

The project includes multiple time-series forecasting approaches.

Run ARIMA:

    python api/models/time_series/arima_model.py

Run SARIMA:

    python api/models/time_series/sarima_model.py

Run Holt-Winters:

    python api/models/time_series/holt_winters_model.py

Run Prophet:

    python api/models/time_series/prophet_model.py

These models provide different approaches for analyzing and forecasting temperature time-series patterns.

---

## 17. Compare Time-Series Models

Run:

    python api/models/time_series/time_series_comparison.py

The comparison evaluates:

- ARIMA
- SARIMA
- Holt-Winters
- Prophet

The generated comparison results are stored under:

    reports/results/

SARIMA achieved the best performance among the implemented time-series models in the current experiments.

---

## 18. Run Satellite Image Analysis

The satellite image dataset is stored in:

    satellite/dataset/

The dataset contains:

    satellite/dataset/clear/
    satellite/dataset/cloudy/

Run the satellite image analysis:

    python reports/satellite/satellite_analysis.py

The analysis generates image-processing outputs including:

    reports/satellite/satellite_grayscale.png
    reports/satellite/satellite_edges.png

The analysis includes basic satellite image properties, grayscale processing, and edge detection.

---

## 19. Train Satellite Cloud Classification CNN

Run:

    python reports/satellite/cnn_satellite_model.py

The CNN classifies satellite images into:

- Clear
- Cloudy

The trained model is saved at:

    api/models/deep_learning/cnn_satellite_cloud_model.keras

Evaluation outputs are generated under:

    reports/satellite/

The current CNN implementation is a proof-of-concept because the available labeled satellite dataset is small.

---

## 20. Run Remote-Sensing Integration

The remote-sensing module integrates satellite image classification with the weather intelligence workflow.

Run:

    python reports/satellite/remote_sensing_integration.py

The generated result is saved at:

    reports/satellite/remote_sensing_integration.json

The result contains:

- Satellite classification
- Confidence score
- Weather-variable interpretation

---

## 21. Run SHAP Explainable AI Analysis

The project includes SHAP-based explainability for the Random Forest temperature prediction workflow.

Run:

    python api/xai/shap_analysis.py

The generated SHAP visualization is saved at:

    reports/results/shap_summary.png

The current SHAP workflow uses a lightweight surrogate Random Forest model trained on a small sample for practical SHAP computation.

---

## 22. Start the FastAPI Application

From the project root, run:

    python -m uvicorn api.app:app --reload --port 8001

The application will be available at:

    http://127.0.0.1:8001/app

Keep the terminal running while using the application.

---

## 23. Check Application Health

Open the following URL in a web browser:

    http://127.0.0.1:8001/health

The health endpoint verifies that the FastAPI application is running correctly.

---

## 24. Open Swagger API Documentation

Open:

    http://127.0.0.1:8001/docs

Swagger provides an interactive interface for testing the FastAPI endpoints.

Important API endpoints include:

    /predict
    /health
    /model-info
    /model-performance
    /analytics-data
    /api-info

---

## 25. Test Weather Prediction

Open the Weather Intelligence application:

    http://127.0.0.1:8001/app

Enter the required weather parameters in the prediction form.

The deployed temperature prediction model is:

    RandomForestRegressor

The application uses 13 weather parameters for the prediction interface.

Click:

    Predict Temperature

The application displays the predicted temperature in degrees Celsius.

---

## 26. Open Analytics Dashboard

Open:

    http://127.0.0.1:8001/analytics

The Analytics dashboard displays prediction-related information and visual analysis generated from the application data.

---

## 27. Open Model Performance Dashboard

Open:

    http://127.0.0.1:8001/model-performance-page

The Model Performance page displays model evaluation information including:

- MAE
- MSE
- RMSE
- R²
- Model name
- Temperature regression information

The deployed model is:

    RandomForestRegressor

---

## 28. Open Prediction History

Open:

    http://127.0.0.1:8001/prediction-history-page

The Prediction History page displays:

- Total predictions
- Latest prediction
- Average prediction
- Highest prediction
- Lowest prediction

Prediction history is stored locally by the application.

---

## 29. Open Weather Research Assistant

Open:

    http://127.0.0.1:8001/research-assistant

The Weather Research Assistant provides an AI-based interface for:

- Weather research questions
- Knowledge retrieval
- Temperature prediction
- AI-assisted weather analysis

---

## 30. Test RAG

Open the Research Assistant and enter a knowledge-based question.

Example:

    What is RAG?

The system retrieves relevant information from the weather knowledge base and generates a grounded response.

The knowledge base is located at:

    api/rag/documents/weather_knowledge_base.txt

---

## 31. Test Semantic Vector Search

The project uses semantic embeddings and FAISS for weather knowledge retrieval.

The semantic vector store is located at:

    api/rag/vector_store/semantic_faiss.index

Supporting files include:

    api/rag/vector_store/semantic_chunks.pkl
    api/rag/vector_store/embedding_model.txt

The embedding model used is:

    all-MiniLM-L6-v2

The semantic retrieval process identifies relevant knowledge-base content based on the meaning of the user query.

---

## 32. Test LangChain Integration

The LangChain RAG implementation is located at:

    api/rag/langchain_rag.py

The workflow integrates:

- Semantic embeddings
- FAISS
- LangChain
- Ollama
- Llama 3.2

The LangChain workflow can be tested through the Weather Research Assistant.

---

## 33. Test LangGraph Agent

The LangGraph weather agent is located at:

    api/agent/langraph_weather_agent.py

The workflow uses graph-based state management for weather research tasks.

The LangGraph workflow can be tested through the corresponding research workflow.

---

## 34. Test Llama 3.2 Local LLM

The project uses a local Llama 3.2 model through Ollama.

Check the available Ollama models:

    ollama list

The required model is:

    llama3.2:3b

If the model is not installed, run:

    ollama pull llama3.2:3b

The local model is used for AI-generated responses in the research workflow.

---

## 35. Test AI Agent Routing

The multi-tool AI agent is implemented in:

    api/agent/tool_calling_agent.py

The agent identifies whether a user request is related to:

- Weather research
- Temperature prediction

Example research query:

    What is RAG?

Example prediction request:

    Predict temperature

The appropriate workflow is selected based on the user request.

---

## 36. Test Agent Memory

The AI agent includes in-memory conversation state management.

The memory implementation is integrated into:

    api/agent/tool_calling_agent.py

The workflow can maintain relevant information across turns during an active conversation session.

---

## 37. Test Tool and Function Calling

The project uses callable tools for weather research and temperature prediction.

The prediction tool accepts the required weather parameters and returns a predicted temperature.

The prediction workflow uses:

    RandomForestRegressor

The prediction result is returned in:

    °C

---

## 38. Test MCP Integration

The MCP server is located at:

    api/mcp/weather_mcp_server.py

The MCP server exposes weather-related tools including:

    predict_weather_temperature
    weather_project_information

The MCP integration allows the AI agent to access weather prediction functionality through MCP tools.

The MCP workflow can be tested using the MCP server and client integration.

---

## 39. Test Full Weather Agent

The complete integrated weather agent is located at:

    api/agent/full_weather_agent.py

The full workflow combines:

- LangGraph
- LangChain
- Semantic FAISS
- Llama 3.2
- MCP
- Weather prediction
- Research workflow

The integrated endpoint is:

    http://127.0.0.1:8001/research-full

A successful prediction request returns the prediction result together with model information.

---

## 40. Docker Configuration

The project includes Docker configuration for container-based deployment.

The main Docker configuration file is:

    Dockerfile

The Docker ignore file is:

    .dockerignore

The Dockerfile is configured to run the FastAPI application using Uvicorn.

Docker has not been locally tested in the current environment because Docker is not installed.

If Docker is installed later, build the image using:

    docker build -t weather-intelligence-ai .

Run the container using:

    docker run -p 8000:8000 weather-intelligence-ai

---

## 41. Project Documentation

Project documentation is stored in:

    docs/

Important documentation files include:

    docs/project_overview.md
    docs/installation_guide.md
    docs/EVALUATION_REPORT.md
    docs/USER_MANUAL.md
    docs/api_documentation.md

These documents describe the project overview, installation procedure, evaluation results, user operation, and API functionality.

---

## 42. Recommended Application Startup

For normal local application use, follow these steps.

### Step 1: Open PowerShell

Open a new PowerShell terminal.

### Step 2: Navigate to the Project

    cd "C:\Users\HP\Documents\Weather-Intelligence-AI"

### Step 3: Activate the Virtual Environment

    .\venv\Scripts\Activate.ps1

### Step 4: Start the FastAPI Application

    python -m uvicorn api.app:app --reload --port 8001

### Step 5: Open the Application

Open:

    http://127.0.0.1:8001/app

### Step 6: Open the Research Assistant

Open:

    http://127.0.0.1:8001/research-assistant

---

## 43. Troubleshooting

### Python is not recognized

Check the Python installation:

    python --version

### Virtual environment activation problem

Run:

    .\venv\Scripts\Activate.ps1

### FastAPI server does not start

The required port may already be in use.

Run the application on another port:

    python -m uvicorn api.app:app --reload --port 8002

Then open:

    http://127.0.0.1:8002/app

### Ollama model is unavailable

Check the installed models:

    ollama list

Install the required model:

    ollama pull llama3.2:3b

### Raw dataset is not found

Verify that the dataset exists at:

    datasets/raw/weather.csv

### Preprocessed dataset is not found

Run:

    python api/preprocessing/preprocessing.py

### Feature-engineered dataset is not found

Run:

    python api/feature_engineering/feature_engineering.py

### Machine learning training data is not found

Run:

    python api/models/prepare_data.py

### Docker command is not recognized

Install Docker Desktop before attempting Docker build or container execution.

---

## 44. Final Local Verification Checklist

Before considering the project locally ready, verify the following:

- [ ] Python environment is activated.
- [ ] Required Python libraries are installed.
- [ ] Raw weather dataset is available.
- [ ] Data preprocessing is completed.
- [ ] EDA outputs are generated.
- [ ] Feature engineering is completed.
- [ ] Machine learning training data is generated.
- [ ] Machine learning models are trained.
- [ ] Machine learning model comparison is generated.
- [ ] Deep learning models are evaluated.
- [ ] Time-series models are evaluated.
- [ ] Satellite image analysis is completed.
- [ ] Satellite CNN model is generated.
- [ ] Remote-sensing integration is completed.
- [ ] SHAP analysis output is generated.
- [ ] FastAPI application starts successfully.
- [ ] Health endpoint works.
- [ ] Swagger API documentation opens.
- [ ] Weather prediction works.
- [ ] Analytics dashboard opens.
- [ ] Model Performance dashboard opens.
- [ ] Prediction History opens.
- [ ] Weather Research Assistant opens.
- [ ] RAG query works.
- [ ] Semantic FAISS retrieval works.
- [ ] LangChain integration works.
- [ ] LangGraph workflow works.
- [ ] Llama 3.2 local model is available.
- [ ] AI agent routing works.
- [ ] Agent memory works.
- [ ] Tool/function calling works.
- [ ] MCP server and tools work.
- [ ] Full Weather Agent works.
- [ ] Docker configuration is present.

### Known Limitations

The satellite CNN is a proof-of-concept because the available labeled satellite dataset is very small.

The current satellite CNN experiment achieved limited validation performance and should not be treated as a production-grade satellite classification model.

The SHAP workflow uses a lightweight surrogate Random Forest model trained on a small sample for practical explainability computation rather than directly explaining the full production Random Forest model.

Docker configuration is prepared but has not been locally tested because Docker is not installed in the current environment.

The current main application interface is implemented using FastAPI with HTML, CSS, and JavaScript. Streamlit/Gradio is not currently used as the main application interface.

LlamaIndex has not been implemented and tested in the current project and therefore is not claimed as completed.

The available weather dataset is used for the current implementation and experimentation; it should not be represented as an exclusive Indian Southwest Monsoon dataset unless an appropriate monsoon-specific dataset is added.

### Conclusion

The Weather Intelligence and Climate Decision Support Platform integrates weather data preprocessing, exploratory data analysis, feature engineering, machine learning, deep learning, time-series forecasting, satellite image analysis, remote sensing, explainable AI, FastAPI services, RAG, semantic search, LangChain, LangGraph, local LLM inference, AI agents, memory, tool calling, and MCP-based weather prediction.

The project can be operated locally using the FastAPI application and the installation and verification procedures described in this guide.