# 🌦️ Weather Intelligence AI

An **AI-powered Weather Intelligence and Climate Decision Support Platform** that combines machine learning, deep learning, time-series forecasting, satellite image analysis, RAG, local LLMs, and AI agents for intelligent weather prediction and research assistance.

---

## 📌 Project Overview

This project is designed to provide an intelligent platform for **weather prediction, climate analysis, and weather-related decision support**.

The system integrates **Machine Learning, Deep Learning, Time-Series Forecasting, Satellite Image Analysis, Explainable AI, Retrieval-Augmented Generation (RAG), Local LLMs, and AI Agents** into a unified weather intelligence platform.

The application provides both **weather prediction capabilities** and an **AI-powered research assistant** for answering weather-related questions using retrieved knowledge.

---

## 🚀 Features

✔️ Weather Data Cleaning & Preprocessing  
✔️ Exploratory Data Analysis (EDA)  
✔️ Advanced Feature Engineering  
✔️ Machine Learning Models  
✔️ Deep Learning Models – LSTM, GRU, CNN-LSTM  
✔️ Time-Series Forecasting – ARIMA, SARIMA, Holt-Winters, Prophet  
✔️ Satellite Image Analysis  
✔️ Random Forest Feature Importance for Explainable AI  
✔️ Model Performance Comparison using MAE, MSE, RMSE & R²  
✔️ FastAPI Prediction Service  
✔️ Streamlit User Interface  
✔️ TF-IDF & FAISS based RAG Pipeline  
✔️ Ollama & Llama 3.2 Local LLM Integration  
✔️ AI-powered Weather Research Assistant  
✔️ AI Agent for Prediction & Knowledge Routing  
✔️ Docker Configuration  

---

## 🛠️ Technologies Used

| Technology | Purpose |
|-----------|--------|
| 🐍 Python | Core programming |
| 🌐 HTML | Web page structure |
| 🎨 CSS | User interface styling |
| ⚡ JavaScript | Frontend functionality |
| 🤖 Scikit-learn | Machine Learning |
| 📈 Linear Regression | Regression model |
| 🌳 Decision Tree | Regression model |
| 🌲 Random Forest | Regression model |
| 🚀 Gradient Boosting | Regression model |
| 🧠 TensorFlow / Keras | Deep Learning |
| 🔄 LSTM / GRU / CNN-LSTM | Weather sequence modelling |
| 📊 ARIMA / SARIMA | Time-Series Forecasting |
| 📉 Holt-Winters / Prophet | Time-Series Forecasting |
| 🔎 TF-IDF | Text retrieval |
| 🗂️ FAISS | Vector similarity search |
| 🦙 Ollama | Local LLM runtime |
| 🤖 Llama 3.2 | Local Language Model |
| 🚀 FastAPI | Prediction API |
| 🎈 Streamlit | Interactive interface |
| 📊 Pandas / NumPy | Data processing |
| 📈 Matplotlib | Data visualization |
| 👁️ OpenCV | Image processing |
| 🐳 Docker | Containerization |
| 🔧 Git / GitHub | Version control |

---

## 📂 Project Structure

```text
Weather-Intelligence-AI/
│
├── api/
│   ├── agent/
│   ├── eda/
│   ├── feature_engineering/
│   ├── models/
│   ├── preprocessing/
│   ├── rag/
│   ├── fastapi_app.py
│   ├── app.py
│   └── model_loader.py
│
├── data/
├── datasets/
│   ├── raw/
│   └── processed/
├── docs/
├── models/
│   ├── ml/
│   ├── deep_learning/
│   └── time_series/
├── notebooks/
├── reports/
│   ├── eda/
│   ├── results/
│   ├── satellite/
│   └── xai/
├── satellite/
│   ├── dataset/
│   └── images/
├── src/
├── ui/
├── Dockerfile
├── .dockerignore
├── requirements.txt
└── prediction_history.json
```

---

## 📸 Screenshots

### 🌦️ Weather Dashboard
![Weather Dashboard](screenshots/dashboard.png)

### 🤖 Weather Research Assistant
![Research Assistant](screenshots/research-assistant.png)

### 📊 Analytics Dashboard
![Analytics](screenshots/analytics.png)

### 🧠 Model Performance
![Model Performance](screenshots/model-performance.png)

### 🔮 Weather Prediction
![Prediction](screenshots/prediction.png)

---

## 📊 Machine Learning Results

The evaluated Machine Learning models produced the following results:

| Model | MAE | RMSE | R² |
|------|------:|------:|------:|
| Linear Regression | 2.7483 | 3.4357 | 0.8713 |
| Decision Tree | 1.6317 | 2.4930 | 0.9322 |
| Random Forest | 1.2231 | 1.6763 | 0.9694 |
| Gradient Boosting | 2.3307 | 2.9207 | 0.9070 |

The Random Forest model achieved an **R² score of approximately 96.94%** among the evaluated Machine Learning models.

---

## 🧠 Deep Learning Models

The project evaluates multiple deep learning architectures for weather-related prediction tasks.

✔️ LSTM  
✔️ GRU  
✔️ CNN-LSTM  

The models are evaluated using:

- MAE
- MSE
- RMSE
- R²

---

## 📈 Time-Series Forecasting

The project includes multiple time-series forecasting approaches:

✔️ ARIMA  
✔️ SARIMA  
✔️ Holt-Winters  
✔️ Prophet  

These models are used for analysing and forecasting weather-related time-series patterns.

---

## 🛰️ Satellite Image Analysis

The platform includes satellite image processing and analysis features such as:

✔️ Grayscale Image Analysis  
✔️ Edge Detection  
✔️ Image Dimensions  
✔️ Channel Information  
✔️ Mean Intensity  
✔️ Intensity Standard Deviation  

---

## 🔍 Explainable AI

The project uses **Random Forest tree-based feature importance** to understand the contribution of input features to model predictions.

Feature importance outputs are stored under:

```text
reports/xai/
```

---

## 🤖 RAG Weather Research Assistant

The project uses a Retrieval-Augmented Generation pipeline to answer weather-related research questions.

```text
Weather Knowledge Base
        ↓
Text Chunking
        ↓
TF-IDF Vectorization
        ↓
FAISS Vector Store
        ↓
Relevant Context Retrieval
        ↓
Llama 3.2 Local LLM
        ↓
Weather Answer
```

The RAG system retrieves relevant weather knowledge before generating an answer using the local **Llama 3.2** model.

---

## 🧩 AI Agent

The AI Agent routes user queries between the **weather prediction system** and the **RAG knowledge assistant**.

### 🔮 Prediction Route

```text
User Query
    ↓
AI Agent
    ↓
Query Classifier
    ↓
FastAPI Prediction API
    ↓
Weather AI Model
    ↓
Temperature Prediction
```

### 📚 Knowledge Route

```text
User Query
    ↓
AI Agent
    ↓
RAG Knowledge Base
    ↓
FAISS Retrieval
    ↓
Llama 3.2
    ↓
Generated Answer
```

---

## 🚀 FastAPI Endpoints

| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET | `/` | API information |
| GET | `/health` | Health check |
| POST | `/predict` | Weather prediction |

---

## ⚙️ Installation & Setup

Follow these steps to run the project locally.

### 1️⃣ Clone the Repository

```powershell
git clone https://github.com/Abhigna13/Weather-Intelligence-AI.git
```

### 2️⃣ Navigate to the Project Folder

```powershell
cd Weather-Intelligence-AI
```

### 3️⃣ Create Virtual Environment

```powershell
python -m venv venv
```

### 4️⃣ Activate Virtual Environment

```powershell
.\venv\Scripts\Activate.ps1
```

### 5️⃣ Install Dependencies

```powershell
pip install -r requirements.txt
```

### 6️⃣ Start FastAPI Server

```powershell
python -m uvicorn api.fastapi_app:app --reload --port 8000
```

### 7️⃣ Open FastAPI Documentation

```text
http://127.0.0.1:8000/docs
```

### 8️⃣ Start Streamlit Interface

Open another terminal and run:

```powershell
.\venv\Scripts\Activate.ps1
python -m streamlit run api\app.py
```

### 9️⃣ Open Streamlit Application

```text
http://localhost:8501
```

### 🔟 Run Weather AI Agent

```powershell
python api\agent\weather_agent.py
```

### 1️⃣1️⃣ Run RAG Weather Assistant

```powershell
python api\rag\rag_assistant.py
```

---

## 🐳 Docker

The project also includes Docker configuration for containerized deployment.

### Build Docker Image

```powershell
docker build -t weather-intelligence-ai .
```

### Run Docker Container

```powershell
docker run -p 8000:8000 weather-intelligence-ai
```

---

## 🎓 Internship Project

**MacroEdtech GenAI Research Internship**

This project was developed as part of the **GenAI Research Internship**, focusing on the integration of Artificial Intelligence, Machine Learning, Generative AI, RAG, and intelligent decision-support systems for weather and climate applications.

---

## 🔮 Future Enhancements

✔️ Transformer-based Embeddings  
✔️ Advanced Satellite Image Classification  
✔️ Advanced Weather Forecasting  
✔️ SHAP / LIME Explainability  
✔️ MCP Integration  
✔️ Advanced AI Agent Workflows  
✔️ Cloud Deployment  
✔️ Real-Time Weather API Integration  

---

## 👩‍💻 Author

**Abhigna**

Artificial Intelligence & Data Science Student  
Weather Intelligence AI Project

---

## ⭐ Support

If you found this project helpful, consider giving it a **⭐ on GitHub**.