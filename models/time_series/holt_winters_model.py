import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset
file_path = "datasets/processed/weather_cleaned.csv"
df = pd.read_csv(file_path)

# Convert date and sort
df["Formatted Date"] = pd.to_datetime(df["Formatted Date"], utc=True)
df = df.sort_values("Formatted Date")

# Temperature series
temperature = df["Temperature (C)"].dropna()

# Use last 5000 observations
temperature = temperature.tail(5000).reset_index(drop=True)

# Train-test split
train_size = int(len(temperature) * 0.8)

train = temperature[:train_size]
test = temperature[train_size:]

print("Training samples:", len(train))
print("Testing samples:", len(test))

# Holt-Winters model
model = ExponentialSmoothing(
    train,
    trend="add",
    seasonal="add",
    seasonal_periods=24
)

model_fit = model.fit(optimized=True)

# Forecast
forecast = model_fit.forecast(len(test))

# Evaluation
mae = mean_absolute_error(test, forecast)
mse = mean_squared_error(test, forecast)
rmse = np.sqrt(mse)
r2 = r2_score(test, forecast)

print("\nHolt-Winters Model Performance")
print("--------------------------------")
print(f"MAE  : {mae:.4f}")
print(f"MSE  : {mse:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R2   : {r2:.4f}")

# Plot
plt.figure(figsize=(12, 5))
plt.plot(test.values, label="Actual")
plt.plot(forecast.values, label="Holt-Winters Predicted")

plt.title("Holt-Winters Temperature Forecast")
plt.xlabel("Time")
plt.ylabel("Temperature (C)")
plt.legend()
plt.tight_layout()

output_path = "reports/results/holt_winters_actual_vs_predicted.png"
plt.savefig(output_path, dpi=300)
plt.show()

print(f"\nGraph saved to: {output_path}")