import os
import pickle
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

# ==========================================
# PROJECT ROOT
# ==========================================

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

# ==========================================
# PATHS
# ==========================================

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "ml",
    "research_random_forest_model.pkl"
)

DATA_PATH = os.path.join(
    BASE_DIR,
    "datasets",
    "processed",
    "X_test_ml.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "reports",
    "results"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_PATH = os.path.join(
    OUTPUT_DIR,
    "shap_summary.png"
)

# ==========================================
# LOAD MODEL
# ==========================================

print("=" * 60)
print("SHAP EXPLAINABLE AI ANALYSIS")
print("=" * 60)

print("\nLoading Random Forest model...")

with open(MODEL_PATH, "rb") as file:
    model = pickle.load(file)

print("Model:", type(model).__name__)
print("Original trees:", len(model.estimators_))

# ==========================================
# LOAD TEST DATA
# ==========================================

print("\nLoading test data...")

X_test = pd.read_csv(DATA_PATH)

print("Test data shape:", X_test.shape)

# ==========================================
# CHECK FEATURES
# ==========================================

if hasattr(model, "feature_names_in_"):

    model_features = list(model.feature_names_in_)

    print("Model features:", len(model_features))

    missing_features = [
        col for col in model_features
        if col not in X_test.columns
    ]

    if missing_features:
        print("\nMissing features:")
        print(missing_features)

        raise ValueError(
            "Test data does not contain all model features."
        )

    X_test = X_test[model_features]

# ==========================================
# SHAP SAMPLE
# ==========================================

sample_size = min(20, len(X_test))

X_sample = X_test.iloc[:sample_size]

print("SHAP sample size:", X_sample.shape)

# ==========================================
# SMALL SHAP MODEL
# ==========================================

print("\nCreating SHAP explanation model...")

shap_model = RandomForestRegressor(
    n_estimators=5,
    random_state=42
)

shap_model.fit(
    X_sample,
    model.predict(X_sample)
)

print("SHAP model trees:", len(shap_model.estimators_))

# ==========================================
# SHAP
# ==========================================

print("\nCalculating SHAP values...")

explainer = shap.TreeExplainer(
    shap_model,
    feature_perturbation="tree_path_dependent"
)

shap_values = explainer(
    X_sample,
    check_additivity=False
)

print("SHAP values calculated successfully.")

# ==========================================
# SUMMARY PLOT
# ==========================================

print("\nGenerating SHAP summary plot...")

shap.summary_plot(
    shap_values,
    X_sample,
    show=False
)

plt.tight_layout()

plt.savefig(
    OUTPUT_PATH,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ==========================================
# COMPLETE
# ==========================================

print("\n" + "=" * 60)
print("SHAP ANALYSIS COMPLETED SUCCESSFULLY")
print("=" * 60)

print("\nSaved file:")
print(OUTPUT_PATH)