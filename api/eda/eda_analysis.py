import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Project root
BASE_DIR = Path(__file__).resolve().parents[2]

# Input and output paths
INPUT_PATH = BASE_DIR / "datasets" / "processed" / "weather_preprocessed.csv"
OUTPUT_DIR = BASE_DIR / "reports" / "eda"

# Create output folder
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load dataset
df = pd.read_csv(INPUT_PATH)

print("=" * 50)
print("WEATHER DATASET - EDA")
print("=" * 50)

# Dataset information
print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns.tolist())

print("\nMissing Values:")
print(df.isnull().sum().sum())

# Descriptive statistics
stats = df.describe()
stats.to_csv(OUTPUT_DIR / "descriptive_statistics.csv")

print("\nDescriptive Statistics:")
print(stats)

# -------------------------------
# 1. Temperature Distribution
# -------------------------------

plt.figure(figsize=(10, 6))
plt.hist(df["Temperature (C)"], bins=50)
plt.xlabel("Temperature (°C)")
plt.ylabel("Frequency")
plt.title("Temperature Distribution")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "temperature_distribution.png")
plt.close()

# -------------------------------
# 2. Humidity Distribution
# -------------------------------

plt.figure(figsize=(10, 6))
plt.hist(df["Humidity"], bins=50)
plt.xlabel("Humidity")
plt.ylabel("Frequency")
plt.title("Humidity Distribution")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "humidity_distribution.png")
plt.close()

# -------------------------------
# 3. Wind Speed Distribution
# -------------------------------

plt.figure(figsize=(10, 6))
plt.hist(df["Wind Speed (km/h)"], bins=50)
plt.xlabel("Wind Speed (km/h)")
plt.ylabel("Frequency")
plt.title("Wind Speed Distribution")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "wind_speed_distribution.png")
plt.close()

# -------------------------------
# 4. Pressure Distribution
# -------------------------------

plt.figure(figsize=(10, 6))
plt.hist(df["Pressure (millibars)"], bins=50)
plt.xlabel("Pressure (millibars)")
plt.ylabel("Frequency")
plt.title("Pressure Distribution")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "pressure_distribution.png")
plt.close()

# -------------------------------
# 5. Correlation Matrix
# -------------------------------

correlation = df.corr(numeric_only=True)

correlation.to_csv(OUTPUT_DIR / "correlation_matrix.csv")

plt.figure(figsize=(14, 10))
plt.imshow(correlation, aspect="auto")
plt.colorbar()

plt.title("Weather Feature Correlation Matrix")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "correlation_matrix.png")
plt.close()

# -------------------------------
# 6. Monthly Temperature Analysis
# -------------------------------

monthly_temperature = (
    df.groupby("Month")["Temperature (C)"]
    .mean()
)

monthly_temperature.to_csv(
    OUTPUT_DIR / "monthly_temperature.csv"
)

plt.figure(figsize=(10, 6))
plt.plot(
    monthly_temperature.index,
    monthly_temperature.values,
    marker="o"
)

plt.xlabel("Month")
plt.ylabel("Average Temperature (°C)")
plt.title("Average Temperature by Month")
plt.xticks(range(1, 13))
plt.grid(True)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "monthly_temperature.png")
plt.close()

print("\nEDA completed successfully.")

print("\nEDA files saved in:")
print(OUTPUT_DIR)