import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import load_model


# --------------------------------------------------
# SETTINGS
# --------------------------------------------------

DATA_PATH = "datasets/processed/weather_features.csv"
OUTPUT_DIR = "reports/results"

os.makedirs(OUTPUT_DIR, exist_ok=True)

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

df = pd.read_csv(DATA_PATH)

df = df.dropna(
    subset=FEATURES + [TARGET]
)

df = df.tail(DATA_LIMIT).reset_index(drop=True)

X = df[FEATURES].values
y = df[TARGET].values.reshape(-1, 1)


# --------------------------------------------------
# SCALE DATA
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

for i in range(SEQUENCE_LENGTH, len(X_scaled)):
    X_seq.append(
        X_scaled[i - SEQUENCE_LENGTH:i]
    )
    y_seq.append(
        y_scaled[i]
    )

X_seq = np.array(X_seq)
y_seq = np.array(y_seq)


# --------------------------------------------------
# TRAIN / TEST SPLIT
# --------------------------------------------------

split_index = int(
    len(X_seq) * 0.8
)

X_test = X_seq[split_index:]
y_test = y_seq[split_index:]


# --------------------------------------------------
# MODEL PATHS
# --------------------------------------------------

models = {
    "LSTM": "models/deep_learning/lstm_temperature_model.keras",

    "GRU": "models/deep_learning/gru_temperature_model.keras",

    "CNN-LSTM": "api/models/deep_learning/cnn_lstm_temperature_model.keras"
}


# --------------------------------------------------
# EVALUATE MODELS
# --------------------------------------------------

results = []

for model_name, model_path in models.items():

    print()
    print("Evaluating:", model_name)

    if not os.path.exists(model_path):
        print("Model file not found. Skipping.")
        continue

    model = load_model(model_path)

    predictions_scaled = model.predict(
        X_test,
        verbose=0
    )

    predictions = scaler_y.inverse_transform(
        predictions_scaled
    )

    actual = scaler_y.inverse_transform(
        y_test
    )

    mae = mean_absolute_error(
        actual,
        predictions
    )

    mse = mean_squared_error(
        actual,
        predictions
    )

    rmse = np.sqrt(mse)

    r2 = r2_score(
        actual,
        predictions
    )

    print(f"MAE  : {mae:.4f}")
    print(f"MSE  : {mse:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"R2   : {r2:.4f}")

    results.append({
        "Model": model_name,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2
    })


# --------------------------------------------------
# CHECK RESULTS
# --------------------------------------------------

if not results:

    print()
    print("No Deep Learning models were found.")
    print("STEP 25 cannot continue until at least")
    print("one trained model file exists.")

    raise SystemExit


# --------------------------------------------------
# CREATE DATAFRAME
# --------------------------------------------------

comparison = pd.DataFrame(results)

print()
print("Deep Learning Model Comparison")
print("===============================")

print(
    comparison.round(4).to_string(
        index=False
    )
)


# --------------------------------------------------
# SAVE CSV
# --------------------------------------------------

csv_path = (
    "reports/results/"
    "deep_learning_model_comparison.csv"
)

comparison.to_csv(
    csv_path,
    index=False
)

print()
print("CSV saved to:")
print(csv_path)


# --------------------------------------------------
# R2 GRAPH
# --------------------------------------------------

plt.figure(figsize=(9, 6))

plt.bar(
    comparison["Model"],
    comparison["R2"] * 100
)

plt.xlabel("Deep Learning Model")
plt.ylabel("R² Score (%)")

plt.title(
    "Deep Learning Model Comparison"
)

plt.ylim(0, 100)

plt.tight_layout()

r2_path = (
    "reports/results/"
    "deep_learning_model_comparison.png"
)

plt.savefig(
    r2_path,
    dpi=300
)

plt.close()

print("R² graph saved to:")
print(r2_path)


# --------------------------------------------------
# RMSE GRAPH
# --------------------------------------------------

plt.figure(figsize=(9, 6))

plt.bar(
    comparison["Model"],
    comparison["RMSE"]
)

plt.xlabel("Deep Learning Model")
plt.ylabel("RMSE")

plt.title(
    "Deep Learning RMSE Comparison"
)

plt.tight_layout()

rmse_path = (
    "reports/results/"
    "deep_learning_rmse_comparison.png"
)

plt.savefig(
    rmse_path,
    dpi=300
)

plt.close()

print("RMSE graph saved to:")
print(rmse_path)


# --------------------------------------------------
# BEST MODEL
# --------------------------------------------------

best_model = comparison.loc[
    comparison["R2"].idxmax(),
    "Model"
]

best_r2 = comparison["R2"].max()

print()
print("Best Deep Learning Model")
print("========================")

print("Model:", best_model)
print(f"R²   : {best_r2:.4f}")

print()
print("completed successfully!")