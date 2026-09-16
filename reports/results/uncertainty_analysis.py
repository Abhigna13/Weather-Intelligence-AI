import json
import joblib
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Hide repeated sklearn feature-name warnings
warnings.filterwarnings(
    "ignore",
    message="X has feature names, but DecisionTreeRegressor was fitted without feature names"
)

# ============================================================
# WEATHER INTELLIGENCE AI
# RANDOM FOREST UNCERTAINTY ANALYSIS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

MODEL_PATH = BASE_DIR / "models" / "best_weather_model.pkl"
HISTORY_PATH = BASE_DIR / "prediction_history.json"
RESULTS_DIR = BASE_DIR / "reports" / "results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# LOAD MODEL
# ------------------------------------------------------------

print("Loading Random Forest model...")

model = joblib.load(MODEL_PATH)

print("Model loaded successfully.")
print(f"Model: {type(model).__name__}")
print(f"Number of trees: {len(model.estimators_)}")


# ------------------------------------------------------------
# EXACT 13 FEATURES
# ------------------------------------------------------------

FEATURES = [
    "Summary",
    "Precip Type",
    "Apparent Temperature (C)",
    "Humidity",
    "Wind Speed (km/h)",
    "Wind Bearing (degrees)",
    "Visibility (km)",
    "Loud Cover",
    "Pressure (millibars)",
    "Daily Summary",
    "Year",
    "Month",
    "Day"
]

API_MAPPING = {
    "Summary": "summary",
    "Precip Type": "precip_type",
    "Apparent Temperature (C)": "apparent_temperature",
    "Humidity": "humidity",
    "Wind Speed (km/h)": "wind_speed",
    "Wind Bearing (degrees)": "wind_bearing",
    "Visibility (km)": "visibility",
    "Loud Cover": "cloud_cover",
    "Pressure (millibars)": "pressure",
    "Daily Summary": "daily_summary",
    "Year": "year",
    "Month": "month",
    "Day": "day"
}


# ------------------------------------------------------------
# LOAD PREDICTION HISTORY
# ------------------------------------------------------------

print("Loading prediction history...")

with open(HISTORY_PATH, "r", encoding="utf-8") as file:
    history = json.load(file)

if isinstance(history, dict):

    if "predictions" in history:
        history = history["predictions"]

    elif "history" in history:
        history = history["history"]


print(f"Prediction history records: {len(history)}")


# ------------------------------------------------------------
# BUILD INPUT MATRIX
# ------------------------------------------------------------

rows = []

for record in history:

    values = record.get("input", record)

    row = []

    complete = True

    for feature in FEATURES:

        api_name = API_MAPPING[feature]

        if api_name not in values:

            complete = False
            break

        row.append(values[api_name])

    if complete:
        rows.append(row)


if not rows:

    raise ValueError(
        "No complete 13-feature prediction records found."
    )


# IMPORTANT:
# Use NumPy array instead of DataFrame.
# This matches how the original DecisionTree estimators
# were trained and removes the feature-name warning.

X = np.asarray(rows, dtype=float)

print(f"Uncertainty samples: {X.shape[0]}")
print(f"Input features: {X.shape[1]}")


# ------------------------------------------------------------
# INDIVIDUAL TREE PREDICTIONS
# ------------------------------------------------------------

print("Calculating predictions from 100 Random Forest trees...")

tree_predictions = np.array([
    tree.predict(X)
    for tree in model.estimators_
])

# Shape:
# samples × trees
tree_predictions = tree_predictions.T

print("Tree predictions calculated successfully.")


# ------------------------------------------------------------
# UNCERTAINTY CALCULATIONS
# ------------------------------------------------------------

mean_prediction = np.mean(
    tree_predictions,
    axis=1
)

std_prediction = np.std(
    tree_predictions,
    axis=1
)

lower_bound = np.percentile(
    tree_predictions,
    5,
    axis=1
)

upper_bound = np.percentile(
    tree_predictions,
    95,
    axis=1
)

interval_width = (
    upper_bound - lower_bound
)


# ------------------------------------------------------------
# RESULT TABLE
# ------------------------------------------------------------

results = pd.DataFrame(
    X,
    columns=FEATURES
)

results["Predicted_Temperature_C"] = mean_prediction
results["Uncertainty_STD_C"] = std_prediction
results["Lower_90_Percent_C"] = lower_bound
results["Upper_90_Percent_C"] = upper_bound
results["Interval_Width_C"] = interval_width


# ------------------------------------------------------------
# SAVE CSV
# ------------------------------------------------------------

csv_path = (
    RESULTS_DIR /
    "uncertainty_analysis.csv"
)

results.to_csv(
    csv_path,
    index=False
)

print(f"Saved: {csv_path}")


# ------------------------------------------------------------
# SUMMARY
# ------------------------------------------------------------

summary = {

    "analysis": "Random Forest ensemble uncertainty",

    "model": "RandomForestRegressor",

    "number_of_trees":
        len(model.estimators_),

    "samples_analyzed":
        len(results),

    "mean_predicted_temperature_c":
        float(np.mean(mean_prediction)),

    "mean_uncertainty_std_c":
        float(np.mean(std_prediction)),

    "mean_90_percent_interval_width_c":
        float(np.mean(interval_width)),

    "minimum_uncertainty_std_c":
        float(np.min(std_prediction)),

    "maximum_uncertainty_std_c":
        float(np.max(std_prediction))
}


summary_path = (
    RESULTS_DIR /
    "uncertainty_summary.json"
)

with open(
    summary_path,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        summary,
        file,
        indent=4
    )

print(f"Saved: {summary_path}")


# ------------------------------------------------------------
# UNCERTAINTY VISUALIZATION
# ------------------------------------------------------------

sample_count = min(
    20,
    len(results)
)

x_axis = np.arange(sample_count)

plt.figure(figsize=(12, 6))

plt.errorbar(
    x_axis,
    mean_prediction[:sample_count],
    yerr=std_prediction[:sample_count],
    fmt="o",
    capsize=5
)

plt.xlabel(
    "Prediction Sample"
)

plt.ylabel(
    "Temperature Prediction (°C)"
)

plt.title(
    "Random Forest Temperature Prediction Uncertainty"
)

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

plot_path = (
    RESULTS_DIR /
    "uncertainty_prediction_intervals.png"
)

plt.savefig(
    plot_path,
    dpi=300
)

plt.close()


print(f"Saved: {plot_path}")


# ------------------------------------------------------------
# FINAL OUTPUT
# ------------------------------------------------------------

print()
print("==============================================")
print("UNCERTAINTY ANALYSIS COMPLETED SUCCESSFULLY")
print("==============================================")
print()

print(
    f"Samples analyzed: {len(results)}"
)

print(
    f"Random Forest trees: {len(model.estimators_)}"
)

print(
    f"Mean uncertainty (STD): "
    f"{np.mean(std_prediction):.4f} °C"
)

print(
    f"Mean 90% interval width: "
    f"{np.mean(interval_width):.4f} °C"
)

print()
print("Generated files:")
print(f"1. {csv_path}")
print(f"2. {summary_path}")
print(f"3. {plot_path}")