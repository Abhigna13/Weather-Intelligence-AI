import joblib
from pathlib import Path


# ============================================================
# WEATHER INTELLIGENCE AI
# Model Loader
# ============================================================


# ------------------------------------------------------------
# PROJECT ROOT
# ------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent


# ------------------------------------------------------------
# MODEL AND FEATURE PATHS
# ------------------------------------------------------------

MODEL_PATH = BASE_DIR / "models" / "best_weather_model.pkl"

FEATURE_PATH = BASE_DIR / "api" / "feature_columns.pkl"


# ------------------------------------------------------------
# LOAD AI MODEL
# ------------------------------------------------------------

def load_model():

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"""
AI model file not found.

Expected location:
{MODEL_PATH}

Please make sure the following file exists:

models/best_weather_model.pkl
"""
        )

    try:

        model = joblib.load(MODEL_PATH)

        if model is None:
            raise ValueError(
                "The AI model file was loaded but returned None."
            )

        print("✓ Weather AI model loaded successfully")
        print(f"✓ Model path: {MODEL_PATH}")

        return model

    except Exception as e:

        raise RuntimeError(
            f"Failed to load the weather AI model: {str(e)}"
        ) from e


# ------------------------------------------------------------
# LOAD FEATURE COLUMNS
# ------------------------------------------------------------

def load_features():

    if not FEATURE_PATH.exists():

        raise FileNotFoundError(
            f"""
Feature configuration file not found.

Expected location:
{FEATURE_PATH}

Please make sure the following file exists:

api/feature_columns.pkl
"""
        )

    try:

        features = joblib.load(FEATURE_PATH)

        if features is None:
            raise ValueError(
                "Feature file was loaded but returned None."
            )

        if not isinstance(features, (list, tuple)):

            try:
                features = list(features)

            except Exception as e:

                raise TypeError(
                    "Feature columns must be a list, tuple, "
                    "or another iterable collection."
                ) from e

        if len(features) == 0:

            raise ValueError(
                "Feature column list is empty."
            )

        print("✓ Feature columns loaded successfully")
        print(f"✓ Number of features: {len(features)}")

        return list(features)

    except Exception as e:

        raise RuntimeError(
            f"Failed to load feature columns: {str(e)}"
        ) from e


# ------------------------------------------------------------
# MODEL STATUS
# ------------------------------------------------------------

def get_model_status():

    return {
        "model_exists": MODEL_PATH.exists(),
        "features_exists": FEATURE_PATH.exists(),
        "model_path": str(MODEL_PATH),
        "feature_path": str(FEATURE_PATH)
    }