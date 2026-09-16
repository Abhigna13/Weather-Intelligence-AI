import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset
file_path = "datasets/processed/weather_cleaned.csv"
df = pd.read_csv(file_path)

# Prepare datetime
df["Formatted Date"] = pd.to_datetime(df["Formatted Date"], utc=True)
df = df.sort_values("Formatted Date")

# Prepare Prophet data
data = df[["Formatted Date", "Temperature (C)"]].dropna()
data = data.rename(columns={
    "Formatted Date": "ds",
    "Temperature (C)": "y"
})

# Use last 5000 observations
data = data.tail(5000).reset_index(drop=True)

# Train-test split
train_size = int(len(data) * 0.8)

train = data.iloc[:train_size]
test = data.iloc[train_size:]

print("Training samples:", len(train))
print("Testing samples:", len(test))

# Create Prophet model
model = Prophet(
    daily_seasonality=True,
    weekly_seasonality=True,
    yearly_seasonality=True
)

model.fit(train)

# Forecast
future = test[["ds"]].copy()
forecast = model.predict(future)

predicted = forecast["yhat"].values
actual = test["y"].values

# Evaluation
mae = mean_absolute_error(actual, predicted)
mse = mean_squared_error(actual, predicted)
rmse = np.sqrt(mse)
r2 = r2_score(actual, predicted)

print("\nProphet Model Performance")
print("-------------------------")
print(f"MAE  : {mae:.4f}")
print(f"MSE  : {mse:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R2   : {r2:.4f}")

# Plot
plt.figure(figsize=(12, 5))
plt.plot(actual, label="Actual")
plt.plot(predicted, label="Prophet Predicted")

plt.title("Prophet Temperature Forecast")
plt.xlabel("Time")
plt.ylabel("Temperature (C)")
plt.legend()
plt.tight_layout()

output_path = "reports/results/prophet_actual_vs_predicted.png"
plt.savefig(output_path, dpi=300)
plt.close()

print(f"\nGraph saved to: {output_path}")