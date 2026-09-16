import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor
)

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# Project root
BASE_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = (
    BASE_DIR
    / "datasets"
    / "processed"
)

REPORT_DIR = (
    BASE_DIR
    / "reports"
    / "results"
)

MODEL_DIR = (
    BASE_DIR
    / "models"
    / "ml"
)

# Create directories
REPORT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------
# Load training and testing data
# ---------------------------------------

X_train = pd.read_csv(DATA_DIR / "X_train_ml.csv")
X_test = pd.read_csv(DATA_DIR / "X_test_ml.csv")

y_train = pd.read_csv(DATA_DIR / "y_train_ml.csv").squeeze()
y_test = pd.read_csv(DATA_DIR / "y_test_ml.csv").squeeze()


print("=" * 60)
print("MULTIPLE ML MODEL TRAINING")
print("=" * 60)

print("\nTraining data:", X_train.shape)
print("Testing data :", X_test.shape)


# ---------------------------------------
# Define models
# ---------------------------------------

models = {

    "Linear Regression":
        LinearRegression(),

    "Decision Tree":
        DecisionTreeRegressor(
            random_state=42,
            max_depth=20
        ),

    "Random Forest":
        RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        ),

    "Gradient Boosting":
        GradientBoostingRegressor(
            n_estimators=100,
            random_state=42
        )
}


# ---------------------------------------
# Train and evaluate
# ---------------------------------------

results = []

for name, model in models.items():

    print("\n" + "-" * 60)
    print("Training:", name)

    model.fit(X_train, y_train)
    # Save trained model
    if name == "Random Forest":
        model_path = MODEL_DIR / "research_random_forest_model.pkl"

        import pickle

        with open(model_path, "wb") as f:
            pickle.dump(model, f)

        print("Random Forest model saved to:")
        print(model_path)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    mse = mean_squared_error(
        y_test,
        predictions
    )

    rmse = np.sqrt(mse)

    r2 = r2_score(
        y_test,
        predictions
    )

    results.append({
        "Model": name,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2
    })

    print("MAE :", round(mae, 4))
    print("MSE :", round(mse, 4))
    print("RMSE:", round(rmse, 4))
    print("R2  :", round(r2, 4))


# ---------------------------------------
# Save comparison results
# ---------------------------------------

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="R2",
    ascending=False
)

results_df.to_csv(
    REPORT_DIR / "ml_model_comparison.csv",
    index=False
)


print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

print(results_df.to_string(index=False))

print("\nResults saved to:")
print(REPORT_DIR / "ml_model_comparison.csv")