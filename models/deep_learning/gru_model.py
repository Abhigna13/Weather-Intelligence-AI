import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler

# Load dataset
file_path = "datasets/processed/weather_features.csv"
df = pd.read_csv(file_path)

# Features and target
features = [
    "Humidity",
    "Wind Speed (km/h)",
    "Wind Bearing (degrees)",
    "Visibility (km)",
    "Pressure (millibars)"
]

target = "Temperature (C)"

data = df[features + [target]].dropna()

# Use last 5000 observations
data = data.tail(5000).reset_index(drop=True)

# Scale data
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

X_train = X[:train_size]
X_test = X[train_size:]
y_train = y[:train_size]
y_test = y[train_size:]

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))
print("Input shape:", X_train.shape)

# Build GRU model
model = tf.keras.Sequential([
    tf.keras.layers.GRU(64, return_sequences=True,
                        input_shape=(X_train.shape[1], X_train.shape[2])),
    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.GRU(32),
    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Dense(16, activation="relu"),
    tf.keras.layers.Dense(1)
])

model.compile(
    optimizer="adam",
    loss="mse",
    metrics=["mae"]
)

# Train
model.fit(
    X_train,
    y_train,
    epochs=10,
    batch_size=64,
    validation_split=0.1,
    verbose=1
)

# Save model
output_path = "api/models/deep_learning/gru_temperature_model.keras"
model.save(output_path)

print("\nGRU model saved successfully!")
print(f"Model path: {output_path}")