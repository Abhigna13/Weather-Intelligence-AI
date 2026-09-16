Weather Intelligence AI

Evaluation Report

1. Introduction

Weather Intelligence AI is an intelligent weather prediction and climate decision support platform developed as part of the Weather Intelligence and Climate Decision Support Platform work.

The system combines Machine Learning, Deep Learning, Time-Series Forecasting, Satellite Image Analysis, Explainable AI, Retrieval-Augmented Generation (RAG), Local Large Language Models, AI Agents, FastAPI, and Streamlit.

The primary objective of the implemented system is to analyze meteorological observations, develop and evaluate multiple prediction models, provide temperature prediction through an application interface, analyze satellite imagery, retrieve weather knowledge, and route user queries through an AI agent.

---

2. Dataset Description

The primary meteorological dataset contains historical weather observations.

The dataset includes the following important attributes:

- Formatted Date
- Summary
- Precipitation Type
- Temperature
- Apparent Temperature
- Humidity
- Wind Speed
- Wind Bearing
- Visibility
- Cloud-related information
- Pressure
- Daily Summary

The original dataset contained 96,453 records.

During data cleaning:

- 517 records with missing precipitation type were removed.
- 24 duplicate records were removed.

The final cleaned dataset contains:

95,912 records and 12 columns.

After preprocessing and feature engineering, the dataset was transformed into a model-ready representation.

---

3. Data Preprocessing

The preprocessing pipeline performs the following operations:

- Missing-value handling
- Duplicate removal
- Numerical feature processing
- Categorical feature encoding
- Date and time feature extraction
- Feature transformation
- Dataset preparation for model training

The preprocessed dataset contains:

95,912 rows and 40 features/columns.

The processed dataset contains no missing values.

The preprocessing output is stored under:

datasets/processed/weather_preprocessed.csv

---

4. Exploratory Data Analysis

Exploratory Data Analysis was performed to understand the statistical and distributional characteristics of the weather dataset.

The analysis includes:

- Descriptive statistics
- Correlation analysis
- Temperature distribution
- Humidity distribution
- Pressure distribution
- Wind-speed distribution
- Monthly temperature analysis

Generated EDA outputs are stored under:

reports/eda/

Important generated files include:

descriptive_statistics.csv
correlation_matrix.csv
correlation_matrix.png
temperature_distribution.png
humidity_distribution.png
pressure_distribution.png
wind_speed_distribution.png
monthly_temperature.csv
monthly_temperature.png

---

5. Feature Engineering

Feature engineering was performed to improve the representation of weather relationships.

The feature engineering pipeline generated additional features including:

- Temperature_Difference
- Wind_Temperature_Interaction
- Humidity_Temperature_Interaction
- Pressure_Temperature_Interaction
- Wind_U
- Wind_V
- Hour_Sin
- Hour_Cos
- Month_Sin
- Month_Cos
- DayOfWeek_Sin
- DayOfWeek_Cos

The final feature-engineered dataset contains:

95,912 rows and 52 columns.

The output is stored at:

datasets/processed/weather_features.csv

---

6. Machine Learning Evaluation

Four Machine Learning regression models were implemented and evaluated:

1. Linear Regression
2. Decision Tree
3. Random Forest
4. Gradient Boosting

The evaluation metrics used were:

- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R² Score

6.1 Machine Learning Results

Model| MAE| MSE| RMSE| R²
Linear Regression| 2.7483| 11.8037| 3.4357| 0.8713
Decision Tree| 1.6317| 6.2153| 2.4930| 0.9322
Random Forest| 1.2231| 2.8100| 1.6763| 0.9694
Gradient Boosting| 2.3307| 8.5307| 2.9207| 0.9070

6.2 Machine Learning Findings

Random Forest achieved the best reported Machine Learning performance.

Performance:

- MAE: 1.2231 °C
- RMSE: 1.6763 °C
- R²: 0.9694
- Explained variance: approximately 96.94%

Therefore, Random Forest was selected as the best-performing reported Machine Learning model.

The model comparison results are stored at:

reports/results/ml_model_comparison.csv

---

7. Deep Learning Evaluation

Deep Learning experiments were performed using sequential weather data.

The implemented models include:

- LSTM
- GRU
- CNN-LSTM

The models were evaluated using:

- MAE
- MSE
- RMSE
- R²

7.1 Deep Learning Results

Model| MAE| MSE| RMSE| R²
LSTM| 4.3072| 31.1911| 5.5849| 0.3015
GRU| 4.7844| 34.4057| 5.8656| 0.2295
CNN-LSTM| 4.7602| 35.0510| 5.9204| 0.2150

7.2 Deep Learning Findings

Among the reported Deep Learning experiments, LSTM achieved the best results.

LSTM performance:

- MAE: 4.3072 °C
- RMSE: 5.5849 °C
- R²: 0.3015

The Deep Learning comparison results are stored at:

reports/results/deep_learning_model_comparison.csv

Visual comparisons are stored at:

reports/results/deep_learning_model_comparison.png
reports/results/deep_learning_rmse_comparison.png

---

8. Time-Series Forecasting Evaluation

Time-Series forecasting experiments were performed using:

- ARIMA
- SARIMA
- Holt-Winters
- Prophet

The models were evaluated using MAE, MSE, RMSE, and R².

8.1 Time-Series Results

Model| MAE| MSE| RMSE| R²
ARIMA| 2.3645| 7.8129| 2.7951| -0.6051
SARIMA| 2.2931| 7.3840| 2.7174| -0.5170
Holt-Winters| 5.2619| 41.8781| 6.4713| -7.6035
Prophet| 24.2037| 901.5134| 30.0252| -184.2085

8.2 Time-Series Findings

SARIMA achieved the best reported results within the evaluated Time-Series experiment.

SARIMA performance:

- MAE: 2.2931
- RMSE: 2.7174
- R²: -0.5170

The negative R² values indicate that these particular time-series configurations did not outperform the corresponding baseline implied by the R² evaluation.

These results should therefore be interpreted as experimental evaluation results rather than evidence that the time-series models are production-ready.

Results are stored at:

reports/results/time_series_model_comparison.csv

---

9. Satellite Image Analysis

Satellite image analysis was implemented using image-processing techniques.

A real satellite image was analyzed using OpenCV-based processing.

The analyzed image had:

- Width: 5587 pixels
- Height: 3378 pixels
- Channels: 3
- Mean Intensity: 128.36
- Standard Deviation of Intensity: 72.73

The system generated:

- Grayscale satellite imagery
- Edge-detected satellite imagery

Generated outputs are stored under:

reports/satellite/

Important outputs include:

satellite_grayscale.png
satellite_edges.png

The current implementation focuses on basic satellite image analysis and preprocessing rather than advanced satellite classification or segmentation.

---

10. Explainable AI

Explainability analysis was performed using Random Forest tree-based feature importance.

The analysis identifies weather features that contribute most strongly to the Random Forest prediction model.

The top reported features were:

Feature| Importance
Month_Cos| 0.429236
Humidity| 0.148618
Month_Sin| 0.125065
Precip Type_snow| 0.101553
Month| 0.068523
Day| 0.021652
Pressure (millibars)| 0.021379
Year| 0.015995
Wind_U| 0.012543
Hour_Cos| 0.011667

The generated explainability files are stored under:

reports/xai/

Important files include:

xai_feature_importance.csv
xai_feature_importance.png

The current implementation uses Random Forest feature importance for explainability.

SHAP/LIME-based explainability is identified as a future enhancement.

---

11. RAG Weather Research Assistant Evaluation

A Weather Knowledge Base was created containing information related to:

- Weather prediction
- Temperature
- Humidity
- Pressure
- Wind
- Precipitation
- Cloud cover
- Machine Learning
- Random Forest
- Deep Learning
- Time-Series Forecasting
- Satellite analysis
- Data preprocessing
- Feature engineering
- Model evaluation
- Explainable AI
- Climate decision support
- RAG
- AI Agents
- Weather research

The knowledge base was divided into text chunks.

The implemented retrieval pipeline uses:

Weather Knowledge Base
        |
        v
Text Chunking
        |
        v
TF-IDF Vectorization
        |
        v
FAISS Vector Store
        |
        v
Relevant Context Retrieval
        |
        v
Llama 3.2 Local LLM
        |
        v
Generated Answer

The vector store contains:

11 knowledge chunks

with an embedding/vector representation of:

898 dimensions.

The implemented retrieval approach uses TF-IDF vectors stored and searched using FAISS.

It does not currently use transformer-based sentence embeddings.

---

12. Local LLM Evaluation

The local Large Language Model integration was implemented using:

- Ollama
- Llama 3.2 3B

The LLM was tested with weather-related research questions.

Example query:

What is Random Forest and how is it used for weather prediction?

The RAG system retrieved relevant knowledge and passed the retrieved context to the local Llama 3.2 model.

The generated response correctly described Random Forest as an ensemble Machine Learning algorithm based on multiple decision trees and explained its use in weather prediction.

The local LLM therefore successfully demonstrated:

- Local model execution
- Retrieved-context integration
- RAG-based answer generation
- Weather-domain question answering

---

13. AI Agent Evaluation

An AI Agent was implemented to route user queries to the appropriate system.

The agent supports two primary routes.

13.1 Prediction Route

User Query
    |
    v
AI Agent
    |
    v
Query Classifier
    |
    v
FastAPI Prediction API
    |
    v
Weather AI Model
    |
    v
Temperature Prediction

Example:

predict temperature

The query was classified as:

prediction

and routed to the FastAPI prediction service.

The prediction route was successfully tested with the weather prediction API.

13.2 Knowledge Route

User Query
    |
    v
AI Agent
    |
    v
RAG Knowledge Base
    |
    v
FAISS Retrieval
    |
    v
Llama 3.2
    |
    v
Generated Answer

Example:

What is Random Forest?

The query was classified as:

rag

and routed to the RAG + LLM system.

The AI Agent therefore demonstrated successful routing between prediction and knowledge-query workflows.

---

14. FastAPI Evaluation

A FastAPI service was developed to expose the weather prediction model through HTTP endpoints.

The implemented endpoints include:

GET /
GET /health
POST /predict

14.1 Health Endpoint

The "/health" endpoint returned:

{
    "status": "healthy",
    "model_loaded": true,
    "feature_count": 13
}

This confirms that:

- The API is running.
- The weather model is loaded.
- The prediction service expects 13 input parameters.

14.2 Prediction Endpoint

The prediction endpoint was tested with:

{
    "values": [
        20,
        0.5,
        10,
        180,
        10,
        7,
        1015,
        0,
        0,
        0,
        0,
        0,
        0
    ]
}

The service successfully returned a temperature prediction response.

This confirms successful integration between:

FastAPI
   |
   v
Model Loader
   |
   v
Weather ML Model
   |
   v
Prediction Response

Swagger documentation is available during local execution at:

http://127.0.0.1:8000/docs

---

15. Streamlit Application Evaluation

The Streamlit interface provides an interactive weather prediction application.

The application supports:

- Weather parameter input
- Temperature prediction
- Model information
- Model performance metrics
- Analytics
- Prediction history
- Latest prediction information
- Average prediction
- Highest prediction
- Lowest prediction
- History clearing

The application uses the trained weather model and provides an accessible interface for users.

The application can be started using:

python -m streamlit run api\app.py

The local application is available at:

http://localhost:8501

---

16. Overall Model Comparison

The final reported model comparison included Machine Learning, Deep Learning, and Time-Series experiments.

16.1 Reported Results

Model| MAE| RMSE| R²
Random Forest| 1.2231| 1.6763| 0.9694
Decision Tree| 1.6317| 2.4930| 0.9322
Gradient Boosting| 2.3307| 2.9207| 0.9070
Linear Regression| 2.7483| 3.4357| 0.8713
LSTM| 4.3072| 5.5849| 0.3015
GRU| 4.7844| 5.8656| 0.2295
CNN-LSTM| 4.7602| 5.9204| 0.2150
ARIMA| 2.3645| 2.7951| -0.6051
SARIMA| 2.2931| 2.7174| -0.5170
Holt-Winters| 5.2619| 6.4713| -7.6035
Prophet| 24.2037| 30.0252| -184.2085

The reported Machine Learning experiments achieved substantially stronger performance than the evaluated Deep Learning and Time-Series configurations.

Random Forest produced the strongest reported result on its evaluation setup:

MAE  = 1.2231
RMSE = 1.6763
R²   = 0.9694

The comparisons should be interpreted carefully because the ML, DL, and Time-Series experiments used different modelling setups and evaluation pipelines.

---

17. Generated Reports and Artifacts

The project generated reports and visualizations under:

reports/

Major directories include:

reports/
├── eda/
├── results/
├── satellite/
└── xai/

Important result files include:

reports/results/ml_model_comparison.csv
reports/results/deep_learning_model_comparison.csv
reports/results/time_series_model_comparison.csv
reports/results/final_model_comparison.csv

Important visualizations include:

reports/results/ml_model_comparison.png
reports/results/deep_learning_model_comparison.png
reports/results/deep_learning_rmse_comparison.png
reports/results/time_series_r2_comparison.png
reports/results/time_series_rmse_comparison.png
reports/results/final_model_r2_comparison.png
reports/results/final_model_rmse_comparison.png

---

18. System Integration Evaluation

The implemented system integrates multiple components:

                 Weather Dataset
                       |
                       v
              Data Preprocessing
                       |
                       v
              Feature Engineering
                       |
          +------------+-------------+
          |            |             |
          v            v             v
         ML            DL       Time-Series
          |            |             |
          +------------+-------------+
                       |
                       v
              Model Evaluation
                       |
                       v
             Weather Prediction
                       |
             +---------+---------+
             |                   |
             v                   v
         FastAPI             Streamlit
             |
             v
          AI Agent
          /       \
         /         \
        v           v
 Prediction       RAG
   API             |
                   v
              FAISS Retrieval
                   |
                   v
              Llama 3.2
                   |
                   v
             Research Answer

The integration demonstrates that predictive Machine Learning services can be combined with knowledge retrieval and local LLM-based question answering.

---

19. Docker Evaluation

A Docker configuration was prepared for the project.

The project contains:

Dockerfile
.dockerignore
requirements.txt

The Dockerfile is configured for the Streamlit application.

However, Docker Desktop was not installed on the development system during implementation.

Therefore:

- Dockerfile creation: Completed
- Docker configuration: Prepared
- Local Docker image build: Not tested
- Local Docker container execution: Not tested

Docker deployment remains a configuration-ready component rather than a locally validated deployment.

---

20. Limitations

The current implementation has several limitations.

20.1 Deep Learning Performance

The evaluated Deep Learning models produced lower R² values than the Machine Learning models in the current experiments.

Further tuning, sequence construction, feature selection, normalization, hyperparameter optimization, and longer training may improve performance.

20.2 Time-Series Performance

The evaluated Time-Series models produced negative R² values.

This indicates that the current experimental configurations require further improvement before production use.

20.3 Satellite Analysis

The current satellite component performs image processing and basic image analysis.

Advanced satellite classification, segmentation, cloud detection, and object detection were not fully implemented in the current version.

20.4 Explainability

The current implementation uses Random Forest feature importance.

SHAP/LIME-based explanations are planned as future enhancements.

20.5 RAG Embeddings

The current RAG implementation uses TF-IDF vectorization with FAISS.

Transformer-based embedding models were not used in the current lightweight implementation.

20.6 Advanced Agent Frameworks

Advanced frameworks such as LangChain, LangGraph, LlamaIndex, and MCP-based integrations are planned future enhancements.

20.7 Docker

Docker configuration was prepared but local Docker image building and container execution were not validated because Docker Desktop was unavailable.

---

21. Future Improvements

Future development can include:

- Transformer-based weather embeddings
- Advanced semantic retrieval
- Larger weather knowledge bases
- ChromaDB or other vector databases
- LangChain integration
- LangGraph workflow orchestration
- LlamaIndex integration
- MCP-based tool integration
- Advanced AI-agent workflows
- SHAP/LIME explainability
- Advanced satellite image classification
- Cloud detection and segmentation
- U-Net based satellite segmentation
- YOLO-based weather/satellite object detection
- Transformer-based weather forecasting
- Improved LSTM/GRU/CNN-LSTM architectures
- Advanced time-series optimization
- Real-time weather API integration
- Cloud deployment
- Full Docker-based multi-service deployment

---

22. Conclusion

The Weather Intelligence AI project successfully demonstrates an integrated Artificial Intelligence platform for weather prediction and weather knowledge assistance.

The implementation includes:

- Weather data cleaning
- Data preprocessing
- Exploratory Data Analysis
- Feature engineering
- Multiple Machine Learning models
- Deep Learning models
- Time-Series forecasting experiments
- Satellite image analysis
- Random Forest feature-importance analysis
- Model performance evaluation
- FastAPI prediction service
- Streamlit application
- TF-IDF + FAISS retrieval
- Local Llama 3.2 integration through Ollama
- RAG-based Weather Research Assistant
- AI Agent routing between prediction and knowledge workflows

Among the reported Machine Learning experiments, Random Forest achieved the strongest performance with:

MAE  = 1.2231 °C
RMSE = 1.6763 °C
R²   = 0.9694

The project demonstrates how predictive Machine Learning, supporting Deep Learning and Time-Series experiments, computer vision-based satellite analysis, explainability, RAG, local LLMs, and AI-agent routing can be combined into a single weather intelligence platform.

The current system provides a strong foundation for future extensions toward advanced climate decision support, real-time weather intelligence, sophisticated RAG systems, and multi-agent AI workflows.