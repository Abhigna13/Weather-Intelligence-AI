import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler

# --------------------------------------------------
# 1. Load dataset
# --------------------------------------------------

file_path = "datasets/processed/weather_features.csv"

df = pd.read_csv(file_path)

print("Dataset loaded successfully")
print("Dataset shape:", df.shape)

# --------------------------------------------------
# 2. Select safe input features
# --------------------------------------------------

features = [
    "Humidity",
    "Wind Speed (km/h)",
    "Wind Bearing (degrees)",
    "Visibility (km)",
    "Pressure (millibars)"
]

target = "Temperature (C)"

# Check required columns
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

print("Clean data shape:", data.shape)

# --------------------------------------------------
# 3. Use manageable dataset size
# --------------------------------------------------

data = data.tail(5000).reset_index(drop=True)

print("Data used for CNN-LSTM:", data.shape)

# --------------------------------------------------
# 4. Scale features
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
# 5. Create sequences
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
print("Target shape:", y.shape)

# --------------------------------------------------
# 6. Train-test split
# --------------------------------------------------

train_size = int(len(X) * 0.8)

X_train = X[:train_size]
X_test = X[train_size:]

y_train = y[:train_size]
y_test = y[train_size:]

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

# --------------------------------------------------
# 7. Build CNN-LSTM model
# --------------------------------------------------

model = tf.keras.Sequential([
    
    tf.keras.layers.Input(
        shape=(
            X_train.shape[1],
            X_train.shape[2]
        )
    ),

    tf.keras.layers.Conv1D(
        filters=64,
        kernel_size=3,
        activation="relu"
    ),

    tf.keras.layers.MaxPooling1D(
        pool_size=2
    ),

    tf.keras.layers.LSTM(
        64
    ),

    tf.keras.layers.Dropout(
        0.2
    ),

    tf.keras.layers.Dense(
        32,
        activation="relu"
    ),

    tf.keras.layers.Dense(
        1
    )
])

# --------------------------------------------------
# 8. Compile
# --------------------------------------------------

model.compile(
    optimizer="adam",
    loss="mse",
    metrics=["mae"]
)

model.summary()

# --------------------------------------------------
# 9. Train
# --------------------------------------------------

print("\nStarting CNN-LSTM training...\n")

model.fit(
    X_train,
    y_train,
    epochs=10,
    batch_size=64,
    validation_split=0.1,
    verbose=1
)

# --------------------------------------------------
# 10. Save model
# --------------------------------------------------

output_path = (
    "api/models/deep_learning/"
    "cnn_lstm_temperature_model.keras"
)

model.save(output_path)

print("\nCNN-LSTM model saved successfully!")
print("Model path:", output_path)