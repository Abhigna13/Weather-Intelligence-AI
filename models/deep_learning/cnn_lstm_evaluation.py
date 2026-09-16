import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# --------------------------------------------------
# 1. Load dataset
# --------------------------------------------------

file_path = "datasets/processed/weather_features.csv"

df = pd.read_csv(file_path)

print("Dataset loaded successfully")
print("Dataset shape:", df.shape)


# --------------------------------------------------
# 2. Features and target
# --------------------------------------------------

features = [
    "Humidity",
    "Wind Speed (km/h)",
    "Wind Bearing (degrees)",
    "Visibility (km)",
    "Pressure (millibars)"
]

target = "Temperature (C)"

required_columns = features + [target]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing columns in dataset: {missing_columns}"
    )

data = df[required_columns].dropna()

# Same 5000 records used during training
data = data.tail(5000).reset_index(drop=True)

print("Evaluation data shape:", data.shape)


# --------------------------------------------------
# 3. Scaling
# --------------------------------------------------

feature_scaler = MinMaxScaler()
target_scaler = MinMaxScaler()

X_scaled = feature_scaler.fit_transform(
    data[features]
)

y_scaled = target_scaler.fit_transform(
    data[[target]]
)


# --------------------------------------------------
# 4. Create sequences
# --------------------------------------------------

sequence_length = 24

X = []
y = []

for i in range(sequence_length, len(data)):
    X.append(
        X_scaled[i - sequence_length:i]
    )
    y.append(
        y_scaled[i]
    )

X = np.array(X)
y = np.array(y)

print("Sequence shape:", X.shape)


# --------------------------------------------------
# 5. Same 80/20 split
# --------------------------------------------------

train_size = int(len(X) * 0.8)

X_test = X[train_size:]
y_test = y[train_size:]

print("Testing samples:", len(X_test))


# --------------------------------------------------
# 6. Load trained CNN-LSTM model
# --------------------------------------------------

model_path = (
    "api/models/deep_learning/"
    "cnn_lstm_temperature_model.keras"
)

model = tf.keras.models.load_model(model_path)

print("CNN-LSTM model loaded successfully")


# --------------------------------------------------
# 7. Prediction
# --------------------------------------------------

predicted_scaled = model.predict(
    X_test,
    verbose=0
)


# --------------------------------------------------
# 8. Convert back to temperature
# --------------------------------------------------

actual = target_scaler.inverse_transform(
    y_test
).flatten()

predicted = target_scaler.inverse_transform(
    predicted_scaled
).flatten()


# --------------------------------------------------
# 9. Calculate metrics
# --------------------------------------------------

mae = mean_absolute_error(
    actual,
    predicted
)

mse = mean_squared_error(
    actual,
    predicted
)

rmse = np.sqrt(mse)

r2 = r2_score(
    actual,
    predicted
)


# --------------------------------------------------
# 10. Display results
# --------------------------------------------------

print("\nCNN-LSTM Model Performance")
print("---------------------------")
print(f"MAE  : {mae:.4f}")
print(f"MSE  : {mse:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R2   : {r2:.4f}")


# --------------------------------------------------
# 11. Save prediction graph
# --------------------------------------------------

plt.figure(figsize=(12, 5))

plt.plot(
    actual,
    label="Actual"
)

plt.plot(
    predicted,
    label="CNN-LSTM Predicted"
)

plt.title(
    "CNN-LSTM Temperature Prediction"
)

plt.xlabel("Time")
plt.ylabel("Temperature (C)")

plt.legend()
plt.tight_layout()

output_path = (
    "reports/results/"
    "cnn_lstm_actual_vs_predicted.png"
)

plt.savefig(
    output_path,
    dpi=300
)

plt.close()

print(
    f"\nGraph saved to: {output_path}"
)