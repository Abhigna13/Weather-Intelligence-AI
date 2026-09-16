from pathlib import Path
import pickle
import pandas as pd
import matplotlib.pyplot as plt

# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = PROJECT_ROOT / "models" / "ml" / "research_random_forest_model.pkl"
DATA_PATH = PROJECT_ROOT / "datasets" / "processed" / "X_test_ml.csv"

OUTPUT_DIR = PROJECT_ROOT / "reports" / "xai"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("Loading Random Forest model...")

with open(MODEL_PATH, "rb") as file:
    model = pickle.load(file)

X_test = pd.read_csv(DATA_PATH)

print(f"Test dataset shape: {X_test.shape}")

# Built-in Random Forest feature importance
print("Calculating feature importance...")

importance = pd.DataFrame({
    "Feature": X_test.columns,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    "Importance",
    ascending=False
)

# Save feature importance data
csv_path = OUTPUT_DIR / "xai_feature_importance.csv"
importance.to_csv(csv_path, index=False)

print(f"Feature importance CSV saved to: {csv_path}")

# Top 15 features
top_features = importance.head(15).sort_values("Importance")

plt.figure(figsize=(10, 7))

plt.barh(
    top_features["Feature"],
    top_features["Importance"]
)

plt.xlabel("Feature Importance")
plt.ylabel("Weather Feature")
plt.title("Random Forest Explainable AI - Top 15 Features")

plt.tight_layout()

graph_path = OUTPUT_DIR / "xai_feature_importance.png"
plt.savefig(graph_path, dpi=300, bbox_inches="tight")
plt.close()

print(f"XAI graph saved to: {graph_path}")

print("\nTop 10 Important Features")
print("==========================")

for _, row in importance.head(10).iterrows():
    print(
        f"{row['Feature']}: "
        f"{row['Importance']:.6f}"
    )

print("\ncompleted successfully!")