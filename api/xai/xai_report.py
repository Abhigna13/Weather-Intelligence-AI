import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

MODEL_PATH = os.path.join(
    BASE_DIR, "models", "ml", "research_random_forest_model.pkl"
)

DATA_PATH = os.path.join(
    BASE_DIR, "datasets", "processed", "X_test_ml.csv"
)

OUTPUT_DIR = os.path.join(BASE_DIR, "reports", "xai")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading Random Forest model...")
model = joblib.load(MODEL_PATH)

print("Loading test data...")
X_test = pd.read_csv(DATA_PATH)

print("Test data shape:", X_test.shape)

importance = model.feature_importances_

result = pd.DataFrame({
    "Feature": X_test.columns,
    "Importance": importance
}).sort_values("Importance", ascending=False)

csv_path = os.path.join(
    OUTPUT_DIR, "feature_importance.csv"
)
result.to_csv(csv_path, index=False)

plt.figure(figsize=(10, 8))
top = result.head(15).sort_values("Importance")

plt.barh(top["Feature"], top["Importance"])
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Random Forest Feature Importance - XAI")
plt.tight_layout()

plot_path = os.path.join(
    OUTPUT_DIR, "feature_importance.png"
)
plt.savefig(plot_path, dpi=300, bbox_inches="tight")
plt.close()

print("\nXAI analysis completed successfully!")
print("CSV saved at:", csv_path)
print("Plot saved at:", plot_path)
