Weather Intelligence AI

User Manual

---

1. Introduction

Weather Intelligence AI is an intelligent weather prediction and climate decision support platform designed to combine machine learning, deep learning, time-series forecasting, satellite image analysis, explainable AI, retrieval-augmented generation, local language models, AI agents, and MCP-based weather tools.

The system provides:

- Weather parameter-based temperature prediction
- Machine Learning model development and comparison
- Deep Learning temperature forecasting experiments
- Time-Series forecasting experiments
- Satellite image analysis
- Satellite cloud-condition classification
- Remote-sensing integration
- Explainable AI analysis
- RAG-based weather knowledge retrieval
- Semantic vector search
- Local LLM-based weather research assistance
- LangChain integration
- LangGraph-based workflow orchestration
- AI Agent query routing
- Tool/function calling
- MCP-based weather prediction tool
- FastAPI-based prediction services
- Interactive HTML/CSS/JavaScript web interface
- Prediction analytics
- Model performance visualization
- Prediction history management

---

2. System Requirements

2.1 Software Requirements

The recommended environment is:

- Windows / Linux / macOS
- Python 3.x
- VS Code
- Git
- Internet connection for initial package/model installation
- Ollama
- Llama 3.2 3B local model
- Optional Docker installation for container deployment

2.2 Python Packages

The project uses libraries including:

- Pandas
- NumPy
- SciPy
- Scikit-learn
- TensorFlow / Keras
- Matplotlib
- OpenCV
- FastAPI
- Uvicorn
- Requests
- FAISS
- Sentence Transformers
- SHAP
- Statsmodels
- Prophet
- LangChain
- LangChain Community
- LangChain Ollama
- LangGraph
- MCP Python SDK
- Ollama Python package

---

3. Project Location

The project root directory is:

C:\Users\HP\Documents\Weather-Intelligence-AI

Open this directory as the main folder in VS Code.

---

4. Project Structure

The major project directories are:

Weather-Intelligence-AI/
│
├── api/
│   ├── agent/
│   ├── eda/
│   ├── feature_engineering/
│   ├── mcp/
│   ├── models/
│   ├── preprocessing/
│   ├── rag/
│   ├── xai/
│   └── frontend/
│
├── datasets/
│   ├── raw/
│   └── processed/
│
├── models/
│   ├── deep_learning/
│   └── time_series/
│
├── reports/
│   ├── eda/
│   ├── results/
│   └── satellite/
│
├── satellite/
│   ├── dataset/
│   └── images/
│
├── docs/
│
├── notebooks/
├── data/
├── src/
│
├── app.py
├── Dockerfile
├── requirements.txt
└── README.md

---

5. Starting the Application

Open PowerShell in:

C:\Users\HP\Documents\Weather-Intelligence-AI

Activate the virtual environment:

.\venv\Scripts\activate

Start the FastAPI application:

python -m uvicorn api.app:app --reload --port 8001

The main application is available at:

http://127.0.0.1:8001/app

The application uses FastAPI as the backend and HTML/CSS/JavaScript for the web interface.

---

6. Health Check

To verify that the API is running, open:

http://127.0.0.1:8001/health

A successful response confirms that the backend is available.

---

7. Swagger API Documentation

FastAPI provides interactive API documentation at:

http://127.0.0.1:8001/docs

Swagger can be used to inspect and test available API endpoints.

---

8. Weather Prediction

8.1 Prediction Interface

Open:

http://127.0.0.1:8001/app

The Weather Intelligence dashboard contains the weather prediction interface.

The prediction system accepts 13 weather-related input parameters.

The parameters include:

1. Temperature
2. Apparent Temperature
3. Humidity
4. Wind Speed
5. Wind Bearing
6. Visibility
7. Pressure
8. Precipitation Type
9. Year
10. Month
11. Day
12. Hour
13. Day of Week

The exact frontend input representation may vary according to the application interface.

---

9. Performing a Prediction

Follow these steps:

1. Open the Weather Intelligence dashboard.
2. Enter the required weather parameters.
3. Verify that all required fields contain valid values.
4. Click the prediction button.
5. Wait for the FastAPI prediction request to complete.
6. The predicted temperature is displayed in degrees Celsius.

The deployed prediction model is:

RandomForestRegressor

The application communicates with the prediction backend through FastAPI.

---

10. Machine Learning Models

The project evaluates multiple machine learning regression models.

The implemented models include:

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting Regressor

The models are evaluated using:

- MAE
- MSE
- RMSE
- R²

The Random Forest model provides the strongest performance among the implemented traditional ML models.

The evaluated Random Forest model achieved approximately:

- MAE: 1.22 °C
- RMSE: 1.68 °C
- R²: 0.9694

The deployed application evaluation summary reports approximately:

- MAE: 1.2434 °C
- RMSE: 1.7324 °C
- R²: 96.73%

These values correspond to different evaluation/reporting stages of the project and should not be treated as contradictory model results.

---

11. Machine Learning Model Comparison

The ML comparison results are stored under:

reports/results/

The comparison output includes model evaluation metrics and visualization files.

The comparison allows users to determine which traditional ML algorithm performs best for temperature prediction.

---

12. Data Preprocessing

The raw weather dataset is located at:

datasets/raw/weather.csv

The preprocessing pipeline:

- Loads the raw dataset
- Handles missing values
- Removes duplicate records
- Converts relevant date/time information
- Encodes categorical variables
- Generates numerical features
- Produces a cleaned and model-ready dataset

The processed dataset is saved under:

datasets/processed/

---

13. Exploratory Data Analysis

EDA can be executed using the project EDA pipeline.

The generated reports are stored under:

reports/eda/

The EDA outputs include:

- Descriptive statistics
- Correlation matrix
- Temperature distribution
- Humidity distribution
- Pressure distribution
- Wind-speed distribution
- Monthly temperature analysis

These visualizations help understand the relationships between weather variables.

---

14. Feature Engineering

Feature engineering generates additional variables to improve prediction capability.

The project includes engineered features such as:

- Temperature difference
- Wind-temperature interaction
- Humidity-temperature interaction
- Pressure-temperature interaction
- Wind U component
- Wind V component
- Hour sine/cosine representation
- Month sine/cosine representation
- Day-of-week sine/cosine representation

The resulting feature dataset is stored under:

datasets/processed/weather_features.csv

---

15. Deep Learning Experiments

The project includes deep learning experiments for weather temperature forecasting.

Implemented models include:

- LSTM
- GRU
- CNN-LSTM

The deep learning models were evaluated using:

- MAE
- MSE
- RMSE
- R²

The generated comparison reports are stored under:

reports/results/

The trained LSTM model is stored under:

models/deep_learning/lstm_temperature_model.keras

---

16. Time-Series Forecasting

The project includes multiple time-series forecasting approaches.

Implemented models include:

- ARIMA
- SARIMA
- Holt-Winters
- Prophet

Evaluation metrics include:

- MAE
- MSE
- RMSE
- R²

The time-series scripts are located under:

models/time_series/

The SARIMA model provided the strongest performance among the implemented time-series approaches.

---

17. Satellite Image Analysis

The satellite analysis component processes satellite imagery for weather-related visual analysis.

The satellite datasets are organized under:

satellite/dataset/

Satellite images are used for:

- Image inspection
- Grayscale conversion
- Edge detection
- Cloud-condition analysis

Generated outputs are stored under:

reports/satellite/

---

18. Satellite Cloud Classification

A CNN-based satellite image classification experiment was implemented for:

- Clear
- Cloudy

The trained model is stored under:

api/models/deep_learning/cnn_satellite_cloud_model.keras

The model was developed as a proof-of-concept using a small dataset.

Because of the limited dataset size, the validation accuracy should not be interpreted as production-level satellite classification performance.

---

19. Remote-Sensing Integration

The remote-sensing integration module connects satellite image analysis with the weather intelligence workflow.

The system can:

1. Load a satellite image.
2. Process the image.
3. Load the trained CNN.
4. Classify the image.
5. Calculate prediction confidence.
6. Interpret the detected cloud condition.
7. Store the integration result.

The generated result is stored under:

reports/satellite/remote_sensing_integration.json

---

20. Explainable AI

The project includes explainability analysis for the Random Forest model.

Two approaches were used during development:

Random Forest Feature Importance

The Random Forest model provides built-in feature importance values that help identify influential input features.

SHAP Analysis

A genuine SHAP-based analysis was also implemented using:

shap.TreeExplainer

The generated SHAP visualization is stored at:

reports/results/shap_summary.png

The SHAP workflow used a small explainability surrogate model because direct SHAP computation on the complete production model was computationally expensive in the development environment.

Therefore, the SHAP result should be interpreted as an explainability experiment rather than a complete production-scale explanation of every prediction.

---

21. Weather Knowledge Base

The weather knowledge base is located at:

api/rag/documents/weather_knowledge_base.txt

It contains information used by the research assistant for grounded weather-related responses.

The knowledge base is divided into retrievable chunks.

---

22. RAG System

The Research Assistant implements Retrieval-Augmented Generation.

The RAG workflow is:

User Question
      ↓
Semantic Search
      ↓
Relevant Knowledge Chunks
      ↓
Context Construction
      ↓
Local LLM
      ↓
Grounded Answer

The system retrieves relevant weather project information before generating the response.

---

23. Semantic Vector Search

The semantic retrieval system uses:

- Sentence Transformers
- FAISS
- "all-MiniLM-L6-v2"

The semantic vector index is stored under:

api/rag/vector_store/

The embedding dimension is 384.

The vector database allows the system to identify knowledge chunks that are semantically related to the user's question.

---

24. LangChain Integration

LangChain is used to connect:

- Semantic retrieval
- FAISS
- Local LLM
- Research workflow

The LangChain RAG implementation is located under:

api/rag/langchain_rag.py

The local LLM used by the research assistant is:

Llama 3.2 3B

---

25. LangGraph Workflow

LangGraph is used for workflow orchestration and agent routing.

The LangGraph implementation is located under:

api/agent/langraph_weather_agent.py

The workflow can route user requests toward appropriate research or prediction functionality.

---

26. AI Agent

The AI Agent accepts natural-language weather questions.

Examples include:

What is RAG?

Which machine learning model is used?

Explain the weather prediction system.

Predict temperature.

The agent determines whether the request requires:

- Research
- Weather prediction
- Tool execution

---

27. Agent Routing

The full weather agent uses conditional routing.

A simplified workflow is:

User Question
      ↓
Query Analysis
      ↓
  ┌───┴────┐
  ↓        ↓
Research Prediction
  ↓        ↓
RAG/LLM    MCP Prediction Tool
  └───┬────┘
      ↓
    Answer

This allows the same assistant interface to support both knowledge-based questions and numerical weather prediction.

---

28. Agent Memory

The agent includes conversation memory using LangGraph workflow state and an in-memory checkpoint mechanism.

This allows relevant previous conversation turns to be retained during an active interaction.

Memory is intended for workflow continuity and conversational context.

---

29. Tool and Function Calling

The AI Agent supports tool-oriented execution.

For prediction requests, the system can route the request to the weather prediction functionality rather than treating the request as a normal knowledge question.

This provides a separation between:

- Knowledge retrieval
- Model prediction
- Tool execution

---

30. MCP Weather Prediction Tool

The project includes an MCP server implemented using the Python MCP SDK.

The MCP server is located at:

api/mcp/weather_mcp_server.py

The server exposes weather-related tools including:

- Temperature prediction
- Weather project information

The prediction tool accepts the required weather parameter values and invokes the trained prediction model.

---

31. Full Weather Agent Integration

The full integration combines:

- Semantic FAISS
- LangChain
- LangGraph
- Llama 3.2 3B
- Weather prediction model
- MCP
- FastAPI

The main full-agent implementation is:

api/agent/full_weather_agent.py

The workflow supports both research and prediction requests.

---

32. Research Assistant

Open:

http://127.0.0.1:8001/research-assistant

The Research Assistant provides:

- Weather research questions
- RAG-based answers
- Semantic retrieval
- Local LLM responses
- Temperature prediction
- AI-agent routing

---

33. Research Question Example

Enter:

What is RAG?

The system retrieves relevant information from the weather knowledge base and generates a grounded response using the local LLM.

---

34. Prediction Through Research Assistant

A prediction request can be made through the Research Assistant.

Example:

Predict temperature

When the required 13 values are supplied, the request is routed to the prediction workflow.

The response includes:

- Predicted temperature
- Unit
- Model name
- Number of features used

---

35. Analytics Page

Open:

http://127.0.0.1:8001/analytics

The Analytics page provides information related to prediction history.

It can display:

- Total predictions
- Latest prediction
- Average prediction
- Highest prediction
- Lowest prediction

The page helps understand the history of predictions generated through the application.

---

36. Model Performance Page

Open:

http://127.0.0.1:8001/model-performance-page

The Model Performance page displays the evaluation information for the deployed model.

Important metrics include:

- MAE
- MSE
- RMSE
- R²

The deployed model is:

RandomForestRegressor

---

37. Prediction History

Open:

http://127.0.0.1:8001/prediction-history-page

The prediction history system records prediction results generated by the application.

Users can review previous prediction results and use the Clear History option when required.

---

38. API Endpoints

Important API routes include:

/health
/predict
/model-info
/model-performance
/analytics-data
/prediction-history
/research
/research-langchain
/langgraph
/research-full
/research-assistant
/api-info

The exact request and response structures can be inspected through:

http://127.0.0.1:8001/docs

---

39. Docker Configuration

A Dockerfile is available in the project root:

Dockerfile

The Docker configuration is prepared for containerized deployment.

However, Docker was not locally executed during the development validation because Docker was not installed/recognized in the development environment.

Therefore:

- Docker configuration: Available
- Dockerfile: Available
- Local Docker execution: Not validated

---

40. Troubleshooting

40.1 Virtual Environment Not Activated

Run:

.\venv\Scripts\activate

---

40.2 Uvicorn Command Not Found

Use:

python -m uvicorn api.app:app --reload --port 8001

---

40.3 Port 8000 Already in Use

Use port 8001:

python -m uvicorn api.app:app --reload --port 8001

Then open:

http://127.0.0.1:8001/app

---

40.4 Ollama Not Running

Start the Ollama application before using the local LLM features.

Verify the model is available using:

ollama list

The required model is:

llama3.2:3b

---

40.5 Research Assistant Not Responding

Check:

1. FastAPI server is running.
2. Ollama is running.
3. Llama 3.2 3B is installed.
4. Semantic FAISS files exist.
5. Weather knowledge base exists.
6. Browser is connected to the correct port.

---

40.6 Prediction Not Working

Check:

1. The Random Forest model exists.
2. Required feature files exist.
3. The FastAPI backend is running.
4. Valid weather parameters are entered.
5. Browser developer console contains no frontend errors.

---

41. Recommended Startup Procedure

For normal application use:

Step 1

Open PowerShell.

Step 2

Navigate to:

C:\Users\HP\Documents\Weather-Intelligence-AI

Step 3

Activate the virtual environment:

.\venv\Scripts\activate

Step 4

Start Ollama if using Research Assistant features.

Step 5

Start FastAPI:

python -m uvicorn api.app:app --reload --port 8001

Step 6

Open:

http://127.0.0.1:8001/app

Step 7

Use the dashboard for prediction or navigate to the Research Assistant.

---

42. Main Application Navigation

The main application provides access to:

- Weather Prediction Dashboard
- Analytics
- Model Performance
- Prediction History
- Research Assistant

The Research Assistant provides access to the GenAI/RAG/agent-based functionality.

---

43. Important Project Limitations

The following limitations should be considered when interpreting the project:

Satellite CNN Dataset

The satellite CNN experiment uses a small proof-of-concept dataset. Its validation performance is therefore not representative of a production satellite classification system.

SHAP Computation

Full SHAP computation on the complete Random Forest model was computationally expensive in the development environment. A small surrogate explainability model was therefore used for the final SHAP visualization.

Docker

Docker configuration is available, but local Docker execution was not validated because Docker was unavailable in the development environment.

Production Deployment

The application is intended as an academic/internship prototype and has not been validated as a production-grade meteorological forecasting service.

Forecast Accuracy

Predictions are model-based estimates and should not be interpreted as official meteorological warnings or safety-critical forecasts.

---

44. User Safety and Interpretation

Weather Intelligence AI should be used as a decision-support and educational system.

Users should not rely exclusively on model predictions for:

- Severe weather warnings
- Emergency decisions
- Disaster response
- Aviation safety
- Marine navigation
- Agricultural decisions involving significant financial risk

Official meteorological sources should be consulted for safety-critical weather information.

---

45. Final Verification Checklist

Before using the application, verify:

- [ ] Virtual environment activated
- [ ] Required packages installed
- [ ] Dataset available
- [ ] Processed datasets available
- [ ] ML models available
- [ ] Deep learning models available
- [ ] Time-series results available
- [ ] Satellite analysis results available
- [ ] CNN satellite model available
- [ ] SHAP result available
- [ ] Semantic FAISS index available
- [ ] Weather knowledge base available
- [ ] Ollama available for local LLM features
- [ ] Llama 3.2 3B available
- [ ] FastAPI starts successfully
- [ ] "/health" works
- [ ] "/docs" works
- [ ] Weather prediction works
- [ ] Analytics page works
- [ ] Model Performance page works
- [ ] Prediction History works
- [ ] Research Assistant works
- [ ] RAG question returns a grounded response
- [ ] Prediction request routes correctly
- [ ] MCP prediction tool works

---

46. Conclusion

Weather Intelligence AI integrates conventional machine learning, deep learning, time-series forecasting, satellite image analysis, remote sensing, explainable AI, RAG, semantic search, local LLMs, LangChain, LangGraph, AI agents,