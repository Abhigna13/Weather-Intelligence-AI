import pandas as pd
import numpy as np
from pathlib import Path


# Project root
BASE_DIR = Path(__file__).resolve().parents[2]

# Input and output paths
INPUT_PATH = (
    BASE_DIR
    / "datasets"
    / "processed"
    / "weather_preprocessed.csv"
)

OUTPUT_PATH = (
    BASE_DIR
    / "datasets"
    / "processed"
    / "weather_features.csv"
)


def create_features():

    print("=" * 50)
    print("WEATHER FEATURE ENGINEERING")
    print("=" * 50)

    # Load preprocessed dataset
    df = pd.read_csv(INPUT_PATH)

    print("\nInput shape:", df.shape)

    # --------------------------------
    # 1. Temperature Difference
    # --------------------------------

    df["Temperature_Difference"] = (
        df["Temperature (C)"]
        - df["Apparent Temperature (C)"]
    )

    # --------------------------------
    # 2. Wind-Temperature Interaction
    # --------------------------------

    df["Wind_Temperature_Interaction"] = (
        df["Wind Speed (km/h)"]
        * df["Temperature (C)"]
    )

    # --------------------------------
    # 3. Humidity-Temperature Interaction
    # --------------------------------

    df["Humidity_Temperature_Interaction"] = (
        df["Humidity"]
        * df["Temperature (C)"]
    )

    # --------------------------------
    # 4. Pressure-Temperature Interaction
    # --------------------------------

    df["Pressure_Temperature_Interaction"] = (
        df["Pressure (millibars)"]
        * df["Temperature (C)"]
    )

    # --------------------------------
    # 5. Wind Components
    # --------------------------------

    wind_direction = np.deg2rad(
        df["Wind Bearing (degrees)"]
    )

    df["Wind_U"] = (
        df["Wind Speed (km/h)"]
        * np.cos(wind_direction)
    )

    df["Wind_V"] = (
        df["Wind Speed (km/h)"]
        * np.sin(wind_direction)
    )

    # --------------------------------
    # 6. Cyclical Hour Features
    # --------------------------------

    df["Hour_Sin"] = np.sin(
        2 * np.pi * df["Hour"] / 24
    )

    df["Hour_Cos"] = np.cos(
        2 * np.pi * df["Hour"] / 24
    )

    # --------------------------------
    # 7. Cyclical Month Features
    # --------------------------------

    df["Month_Sin"] = np.sin(
        2 * np.pi * df["Month"] / 12
    )

    df["Month_Cos"] = np.cos(
        2 * np.pi * df["Month"] / 12
    )

    # --------------------------------
    # 8. Cyclical Day-of-Week Features
    # --------------------------------

    df["DayOfWeek_Sin"] = np.sin(
        2 * np.pi * df["DayOfWeek"] / 7
    )

    df["DayOfWeek_Cos"] = np.cos(
        2 * np.pi * df["DayOfWeek"] / 7
    )

    # Remove infinite values if any
    df = df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # Remove rows containing NaN
    df = df.dropna()

    # Save feature-engineered dataset
    df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print("\nOutput shape:", df.shape)

    print(
        "Missing values:",
        df.isnull().sum().sum()
    )

    print("\nNew features created:")

    new_features = [
        "Temperature_Difference",
        "Wind_Temperature_Interaction",
        "Humidity_Temperature_Interaction",
        "Pressure_Temperature_Interaction",
        "Wind_U",
        "Wind_V",
        "Hour_Sin",
        "Hour_Cos",
        "Month_Sin",
        "Month_Cos",
        "DayOfWeek_Sin",
        "DayOfWeek_Cos"
    ]

    for feature in new_features:
        print(" -", feature)

    print("\nSaved to:")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    create_features()