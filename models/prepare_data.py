import pandas as pd
from sklearn.model_selection import train_test_split

INPUT_FILE = "datasets/processed/weather_features.csv"
OUTPUT_DIR = "datasets/processed"

df = pd.read_csv(INPUT_FILE)

target = "Temperature (C)"

# Remove target and target-derived / leakage features
leakage_columns = [
    "Temperature (C)",
    "Temperature_Difference",
    "Wind_Temperature_Interaction",
    "Humidity_Temperature_Interaction",
    "Pressure_Temperature_Interaction",
    "Apparent Temperature (C)"
]

X = df.drop(columns=leakage_columns, errors="ignore")
y = df[target]

print("Features:", X.shape[1])
print("Samples:", len(X))

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

X_train.to_csv(f"{OUTPUT_DIR}/X_train_ml.csv", index=False)
X_test.to_csv(f"{OUTPUT_DIR}/X_test_ml.csv", index=False)
y_train.to_csv(f"{OUTPUT_DIR}/y_train_ml.csv", index=False)
y_test.to_csv(f"{OUTPUT_DIR}/y_test_ml.csv", index=False)

print("\nData split completed successfully!")
print("X_train:", X_train.shape)
print("X_test :", X_test.shape)
print("y_train:", y_train.shape)
print("y_test :", y_test.shape)