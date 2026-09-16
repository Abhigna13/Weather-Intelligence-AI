import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


INPUT_FILE = "datasets/processed/weather_features.csv"
MODEL_FILE = "api/models/deep_learning/lstm_temperature_model.keras"

df = pd.read_csv(INPUT_FILE)

features = [
    "Humidity",
    "Wind Speed (km/h)",
    "Wind Bearing (degrees)",
    "Visibility (km)",
    "Pressure (millibars)"
]

target = "Temperature (C)"

data = df[features + [target]].dropna()

X = data[features].values
y = data[target].values

# Scaling
X_scaler = MinMaxScaler()
y_scaler = MinMaxScaler()

X_scaled = X_scaler.fit_transform(X)
y_scaled = y_scaler.fit_transform(y.reshape(-1, 1))

# Create sequences
sequence_length = 24

X_sequences = []
y_sequences = []

for i in range(sequence_length, len(X_scaled)):
    X_sequences.append(X_scaled[i-sequence_length:i])
    y_sequences.append(y_scaled[i])

X_sequences = np.array(X_sequences)
y_sequences = np.array(y_sequences)

# Same 80/20 split
split = int(len(X_sequences) * 0.8)

X_test = X_sequences[split:]
y_test = y_sequences[split:]

# Load trained model
model = load_model(MODEL_FILE)

# Prediction
pred_scaled = model.predict(X_test, verbose=0)

pred = y_scaler.inverse_transform(pred_scaled)
actual = y_scaler.inverse_transform(y_test)

# Metrics
mae = mean_absolute_error(actual, pred)
mse = mean_squared_error(actual, pred)
rmse = np.sqrt(mse)
r2 = r2_score(actual, pred)

print("=" * 60)
print("LSTM MODEL EVALUATION")
print("=" * 60)

print(f"MAE : {mae:.4f}")
print(f"MSE : {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R2  : {r2:.4f}")

# Actual vs Predicted graph
plt.figure(figsize=(12, 6))

plt.plot(actual[:300], label="Actual Temperature")
plt.plot(pred[:300], label="Predicted Temperature")

plt.title("LSTM - Actual vs Predicted Temperature")
plt.xlabel("Time Steps")
plt.ylabel("Temperature (°C)")
plt.legend()

plt.tight_layout()

plt.savefig(
    "reports/results/lstm_actual_vs_predicted.png",
    dpi=300
)

plt.show()

print("\nGraph saved successfully!")
print(
    "reports/results/lstm_actual_vs_predicted.png"
)