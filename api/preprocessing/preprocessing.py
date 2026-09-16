import pandas as pd
from pathlib import Path


INPUT_PATH = Path("datasets/processed/weather_cleaned.csv")
OUTPUT_PATH = Path("datasets/processed/weather_preprocessed.csv")


def preprocess_weather_data():

    # Load cleaned dataset
    df = pd.read_csv(INPUT_PATH)

    print("Original shape:", df.shape)

    # Convert date column
    df["Formatted Date"] = pd.to_datetime(
        df["Formatted Date"],
        errors="coerce",
        utc=True
    )

    # Extract useful date/time features
    df["Year"] = df["Formatted Date"].dt.year
    df["Month"] = df["Formatted Date"].dt.month
    df["Day"] = df["Formatted Date"].dt.day
    df["Hour"] = df["Formatted Date"].dt.hour
    df["DayOfWeek"] = df["Formatted Date"].dt.dayofweek

    # Remove rows where date conversion failed
    df = df.dropna(subset=["Formatted Date"])

    # Remove text columns that are not directly required for numerical modeling
    df = df.drop(
        columns=["Formatted Date", "Daily Summary"],
        errors="ignore"
    )

    # Convert categorical columns to numerical values
    categorical_columns = [
        "Summary",
        "Precip Type"
    ]

    df = pd.get_dummies(
        df,
        columns=categorical_columns,
        drop_first=True
    )

    # Make sure all columns are numeric
    df = df.apply(pd.to_numeric, errors="coerce")

    # Remove any remaining missing values
    df = df.dropna()

    # Save preprocessed dataset
    df.to_csv(OUTPUT_PATH, index=False)

    print("Preprocessed shape:", df.shape)
    print("Missing values:", df.isnull().sum().sum())
    print("Saved to:", OUTPUT_PATH)


if __name__ == "__main__":
    preprocess_weather_data()