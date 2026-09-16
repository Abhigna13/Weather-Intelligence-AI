import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset
file_path = "datasets/processed/weather_features.csv"
df = pd.read_csv(file_path)

features = [
    "Humidity",
    "Wind Speed (km/h)",
    "Wind Bearing (degrees)",
    "Visibility (km)",
    "Pressure (millibars)"
]

target = "Temperature (C)"

data = df[features + [target]].dropna()
data = data.tail(5000).reset_index(drop=True)

# Scale
feature_scaler = MinMaxScaler()
target_scaler = MinMaxScaler()

X_scaled = feature_scaler.fit_transform(data[features])
y_scaled = target_scaler.fit_transform(data[[target]])

# Create sequences
sequence_length = 24

X = []
y = []

for i in range(sequence_length, len(data)):
    X.append(X_scaled[i-sequence_length:i])
    y.append(y_scaled[i])

X = np.array(X)
y = np.array(y)

# Train-test split
train_size = int(len(X) * 0.8)

X_test = X[train_size:]
y_test = y[train_size:]

# Load trained model
model = tf.keras.models.load_model(
    "api/models/deep_learning/gru_temperature_model.keras"
)

# Predict
predicted_scaled = model.predict(X_test, verbose=0)

# Convert back to temperature
actual = target_scaler.inverse_transform(y_test)
predicted = target_scaler.inverse_transform(predicted_scaled)

# Metrics
mae = mean_absolute_error(actual, predicted)
mse = mean_squared_error(actual, predicted)
rmse = np.sqrt(mse)
r2 = r2_score(actual, predicted)

print("\nGRU Model Performance")
print("---------------------")
print(f"MAE  : {mae:.4f}")
print(f"MSE  : {mse:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R2   : {r2:.4f}")

# Plot
plt.figure(figsize=(12, 5))
plt.plot(actual, label="Actual")
plt.plot(predicted, label="GRU Predicted")

plt.title("GRU Temperature Prediction")
plt.xlabel("Time")
plt.ylabel("Temperature (C)")
plt.legend()
plt.tight_layout()

output_path = "reports/results/gru_actual_vs_predicted.png"
plt.savefig(output_path, dpi=300)
plt.show()

print(f"\nGraph saved to: {output_path}")