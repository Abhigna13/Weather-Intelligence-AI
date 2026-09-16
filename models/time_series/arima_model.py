import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Project root
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

# Input dataset
DATA_PATH = os.path.join(
    PROJECT_ROOT,
    "datasets",
    "processed",
    "weather_cleaned.csv"
)

# Output directory
OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "reports",
    "results"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# Load dataset
df = pd.read_csv(DATA_PATH)

# Convert date column
df["Formatted Date"] = pd.to_datetime(
    df["Formatted Date"],
    errors="coerce", utc=True
)

df = df.dropna(subset=["Formatted Date"])

# Sort chronologically
df = df.sort_values("Formatted Date")

# Temperature time series
temperature = df["Temperature (C)"].astype(float)

# Use recent 5000 observations
temperature = temperature.tail(5000).reset_index(drop=True)

print("Dataset size:", len(temperature))


# Train-test split
train_size = int(len(temperature) * 0.8)

train = temperature[:train_size]
test = temperature[train_size:]


print("Training samples:", len(train))
print("Testing samples:", len(test))


# ARIMA model
print("\nTraining ARIMA model...")

model = ARIMA(
    train,
    order=(5, 1, 0)
)

model_fit = model.fit()


# Forecast
forecast = model_fit.forecast(
    steps=len(test)
)

forecast = np.asarray(forecast)


# Evaluation
mae = mean_absolute_error(test, forecast)
mse = mean_squared_error(test, forecast)
rmse = np.sqrt(mse)
r2 = r2_score(test, forecast)

print("\nARIMA Model Evaluation")
print("======================")
print(f"MAE  : {mae:.4f}")
print(f"MSE  : {mse:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R2   : {r2:.4f}")


# Save results
results = pd.DataFrame({
    "Actual": test.values,
    "Predicted": forecast
})

results_path = os.path.join(
    OUTPUT_DIR,
    "arima_predictions.csv"
)

results.to_csv(results_path, index=False)


# Plot
plt.figure(figsize=(12, 6))

plt.plot(
    test.values,
    label="Actual Temperature"
)

plt.plot(
    forecast,
    label="ARIMA Forecast"
)

plt.title("ARIMA Temperature Forecast")
plt.xlabel("Time")
plt.ylabel("Temperature (°C)")
plt.legend()
plt.tight_layout()

plot_path = os.path.join(
    OUTPUT_DIR,
    "arima_actual_vs_predicted.png"
)

plt.savefig(plot_path)
plt.close()


print("\nResults saved to:")
print(results_path)

print("Graph saved to:")
print(plot_path)

print("\ncompleted successfully!")