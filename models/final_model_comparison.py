import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "reports" / "results"

# Load comparison files
ml = pd.read_csv(RESULTS_DIR / "ml_model_comparison.csv")
dl = pd.read_csv(RESULTS_DIR / "deep_learning_model_comparison.csv")
ts = pd.read_csv(RESULTS_DIR / "time_series_model_comparison.csv")

# Add model category
ml["Category"] = "Machine Learning"
dl["Category"] = "Deep Learning"
ts["Category"] = "Time Series"

# Combine all results
final_df = pd.concat([ml, dl, ts], ignore_index=True)

# Standardize column names
final_df.columns = [col.strip() for col in final_df.columns]

# Save final comparison
output_csv = RESULTS_DIR / "final_model_comparison.csv"
final_df.to_csv(output_csv, index=False)

print("\nFINAL MODEL COMPARISON")
print("======================")
print(final_df.to_string(index=False))

# Best models based on R2
best_ml = ml.loc[ml["R2"].idxmax()]
best_dl = dl.loc[dl["R2"].idxmax()]
best_ts = ts.loc[ts["R2"].idxmax()]

print("\nBEST MODELS BY CATEGORY")
print("=======================")

print(
    f"Machine Learning : {best_ml['Model']} "
    f"(R2 = {best_ml['R2']:.4f})"
)

print(
    f"Deep Learning    : {best_dl['Model']} "
    f"(R2 = {best_dl['R2']:.4f})"
)

print(
    f"Time Series      : {best_ts['Model']} "
    f"(R2 = {best_ts['R2']:.4f})"
)

# Overall highest R2 among reported results
best_overall = final_df.loc[final_df["R2"].idxmax()]

print("\nBEST REPORTED MODEL")
print("===================")
print(f"Model    : {best_overall['Model']}")
print(f"Category : {best_overall['Category']}")
print(f"MAE      : {best_overall['MAE']:.4f}")
print(f"RMSE     : {best_overall['RMSE']:.4f}")
print(f"R2       : {best_overall['R2']:.4f}")

# R2 comparison graph
plt.figure(figsize=(12, 6))
plt.bar(final_df["Model"], final_df["R2"])
plt.xticks(rotation=45, ha="right")
plt.ylabel("R² Score")
plt.title("Final Model Comparison - R² Score")
plt.tight_layout()

r2_graph = RESULTS_DIR / "final_model_r2_comparison.png"
plt.savefig(r2_graph, dpi=300)
plt.close()

# RMSE comparison graph
plt.figure(figsize=(12, 6))
plt.bar(final_df["Model"], final_df["RMSE"])
plt.xticks(rotation=45, ha="right")
plt.ylabel("RMSE")
plt.title("Final Model Comparison - RMSE")
plt.tight_layout()

rmse_graph = RESULTS_DIR / "final_model_rmse_comparison.png"
plt.savefig(rmse_graph, dpi=300)
plt.close()

print("\nFiles generated:")
print(output_csv)
print(r2_graph)
print(rmse_graph)

print("\ncompleted successfully!")