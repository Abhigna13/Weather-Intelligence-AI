import os
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

RESULTS_DIR = os.path.join(PROJECT_ROOT, "reports", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Actual evaluation results obtained from the four trained models
results = {
    "ARIMA": {
        "MAE": 2.3645,
        "MSE": 7.8129,
        "RMSE": 2.7951,
        "R2": -0.6051
    },
    "SARIMA": {
        "MAE": 2.2931,
        "MSE": 7.3840,
        "RMSE": 2.7174,
        "R2": -0.5170
    },
    "Holt-Winters": {
        "MAE": 5.2619,
        "MSE": 41.8781,
        "RMSE": 6.4713,
        "R2": -7.6035
    },
    "Prophet": {
        "MAE": 24.2037,
        "MSE": 901.5134,
        "RMSE": 30.0252,
        "R2": -184.2085
    }
}

df = pd.DataFrame(results).T
df.index.name = "Model"

print("\nTime-Series Model Comparison")
print("=" * 50)
print(df.round(4))

# Save comparison CSV
csv_path = os.path.join(
    RESULTS_DIR,
    "time_series_model_comparison.csv"
)
df.to_csv(csv_path)

print(f"\nCSV saved to:\n{csv_path}")

# R2 comparison
plt.figure(figsize=(9, 5))
plt.bar(df.index, df["R2"])
plt.xlabel("Model")
plt.ylabel("R² Score")
plt.title("Time-Series Model Comparison - R²")
plt.xticks(rotation=15)
plt.tight_layout()

r2_path = os.path.join(
    RESULTS_DIR,
    "time_series_r2_comparison.png"
)
plt.savefig(r2_path, dpi=300)
plt.close()

print(f"R² graph saved to:\n{r2_path}")

# RMSE comparison
plt.figure(figsize=(9, 5))
plt.bar(df.index, df["RMSE"])
plt.xlabel("Model")
plt.ylabel("RMSE")
plt.title("Time-Series Model Comparison - RMSE")
plt.xticks(rotation=15)
plt.tight_layout()

rmse_path = os.path.join(
    RESULTS_DIR,
    "time_series_rmse_comparison.png"
)
plt.savefig(rmse_path, dpi=300)
plt.close()

print(f"RMSE graph saved to:\n{rmse_path}")

# Best models
best_mae = df["MAE"].idxmin()
best_rmse = df["RMSE"].idxmin()
best_r2 = df["R2"].idxmax()

print("\nBest Time-Series Models")
print("=" * 50)
print(f"Best MAE  : {best_mae} ({df.loc[best_mae, 'MAE']:.4f})")
print(f"Best RMSE : {best_rmse} ({df.loc[best_rmse, 'RMSE']:.4f})")
print(f"Best R²   : {best_r2} ({df.loc[best_r2, 'R2']:.4f})")

print("\ncompleted successfully!")