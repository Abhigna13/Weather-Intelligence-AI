import os
import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout


# --------------------------------------------------
# SETTINGS
# --------------------------------------------------

DATA_PATH = "datasets/processed/weather_features.csv"

MODEL_PATH = (
    "api/models/deep_learning/"
    "lstm_temperature_model.keras"
)

FEATURES = [
    "Humidity",
    "Wind Speed (km/h)",
    "Wind Bearing (degrees)",
    "Visibility (km)",
    "Pressure (millibars)"
]

TARGET = "Temperature (C)"

SEQUENCE_LENGTH = 24
DATA_LIMIT = 5000


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

print("Loading weather dataset...")

df = pd.read_csv(DATA_PATH)

df = df.dropna(
    subset=FEATURES + [TARGET]
)

df = df.tail(DATA_LIMIT).reset_index(drop=True)

print("Dataset shape:", df.shape)


# --------------------------------------------------
# PREPARE FEATURES
# --------------------------------------------------

X = df[FEATURES].values
y = df[TARGET].values.reshape(-1, 1)


# --------------------------------------------------
# SCALE
# --------------------------------------------------

scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)


# --------------------------------------------------
# CREATE SEQUENCES
# --------------------------------------------------

X_seq = []
y_seq = []

for i in range(
    SEQUENCE_LENGTH,
    len(X_scaled)
):
    X_seq.append(
        X_scaled[
            i - SEQUENCE_LENGTH:i
        ]
    )

    y_seq.append(
        y_scaled[i]
    )

X_seq = np.array(X_seq)
y_seq = np.array(y_seq)

print("Sequence shape:", X_seq.shape)


# --------------------------------------------------
# TRAIN / TEST SPLIT
# --------------------------------------------------

split_index = int(
    len(X_seq) * 0.8
)

X_train = X_seq[:split_index]
X_test = X_seq[split_index:]

y_train = y_seq[:split_index]
y_test = y_seq[split_index:]

print("Training samples:", len(X_train))
print("Testing samples :", len(X_test))


# --------------------------------------------------
# BUILD LSTM MODEL
# --------------------------------------------------

model = Sequential([

    LSTM(
        64,
        return_sequences=True,
        input_shape=(
            SEQUENCE_LENGTH,
            len(FEATURES)
        )
    ),

    Dropout(0.2),

    LSTM(
        32,
        return_sequences=False
    ),

    Dropout(0.2),

    Dense(16, activation="relu"),

    Dense(1)
])


# --------------------------------------------------
# COMPILE
# --------------------------------------------------

model.compile(
    optimizer="adam",
    loss="mse",
    metrics=["mae"]
)


# --------------------------------------------------
# TRAIN
# --------------------------------------------------

print()
print("Training LSTM model...")

model.fit(
    X_train,
    y_train,
    epochs=10,
    batch_size=64,
    validation_split=0.1,
    verbose=1
)


# --------------------------------------------------
# SAVE MODEL
# --------------------------------------------------

os.makedirs(
    os.path.dirname(MODEL_PATH),
    exist_ok=True
)

model.save(MODEL_PATH)

print()
print("LSTM model saved successfully!")
print("Model path:")
print(MODEL_PATH)