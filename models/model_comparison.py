import os
import pandas as pd
import matplotlib.pyplot as plt


# --------------------------------------------------
# 1. Actual ML model results
# --------------------------------------------------

results = {
    "Linear Regression": {
        "MAE": 2.7483,
        "MSE": 11.8037,
        "RMSE": 3.4357,
        "R2": 0.8713
    },

    "Decision Tree": {
        "MAE": 1.6317,
        "MSE": 6.2153,
        "RMSE": 2.4930,
        "R2": 0.9322
    },

    "Random Forest": {
        "MAE": 1.2231,
        "MSE": 2.8100,
        "RMSE": 1.6763,
        "R2": 0.9694
    },

    "Gradient Boosting": {
        "MAE": 2.3307,
        "MSE": 8.5307,
        "RMSE": 2.9207,
        "R2": 0.9070
    }
}


# --------------------------------------------------
# 2. Create DataFrame
# --------------------------------------------------

comparison = pd.DataFrame(results).T

print("\nML Model Comparison")
print("===================")

print(comparison.round(4))


# --------------------------------------------------
# 3. Create output directory
# --------------------------------------------------

output_dir = "reports/results"

os.makedirs(
    output_dir,
    exist_ok=True
)


# --------------------------------------------------
# 4. Save comparison CSV
# --------------------------------------------------

csv_path = (
    "reports/results/"
    "ml_model_comparison_final.csv"
)

comparison.to_csv(
    csv_path
)

print()
print("CSV saved to:")
print(csv_path)


# --------------------------------------------------
# 5. R2 comparison
# --------------------------------------------------

plt.figure(figsize=(10, 6))

plt.bar(
    comparison.index,
    comparison["R2"] * 100
)

plt.ylabel("R² Score (%)")
plt.xlabel("Machine Learning Model")

plt.title(
    "Machine Learning Model Comparison"
)

plt.ylim(0, 100)

plt.xticks(
    rotation=20
)

plt.tight_layout()


r2_path = (
    "reports/results/"
    "ml_model_comparison_final.png"
)

plt.savefig(
    r2_path,
    dpi=300
)

plt.close()

print("R² graph saved to:")
print(r2_path)


# --------------------------------------------------
# 6. RMSE comparison
# --------------------------------------------------

plt.figure(figsize=(10, 6))

plt.bar(
    comparison.index,
    comparison["RMSE"]
)

plt.ylabel("RMSE")
plt.xlabel("Machine Learning Model")

plt.title(
    "RMSE Comparison of ML Models"
)

plt.xticks(
    rotation=20
)

plt.tight_layout()


rmse_path = (
    "reports/results/"
    "ml_rmse_comparison.png"
)

plt.savefig(
    rmse_path,
    dpi=300
)

plt.close()

print("RMSE graph saved to:")
print(rmse_path)


# --------------------------------------------------
# 7. Best model
# --------------------------------------------------

best_model = comparison["R2"].idxmax()

best_r2 = comparison.loc[
    best_model,
    "R2"
]

print()
print("Best ML Model")
print("-------------")
print("Model:", best_model)
print(f"R²   : {best_r2:.4f}")

print()
print("STEP 24 completed successfully!")