import os
import json
import numpy as np
import tensorflow as tf

from tensorflow.keras.preprocessing import image


# ============================================================
# PATHS
# ============================================================

MODEL_PATH = (
    "api/models/deep_learning/"
    "cnn_satellite_cloud_model.keras"
)

SATELLITE_IMAGE = (
    "satellite/images/real_satellite.jpg"
)

OUTPUT_DIR = "reports/satellite"

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "remote_sensing_integration.json"
)


# ============================================================
# CHECK FILES
# ============================================================

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"CNN model not found: {MODEL_PATH}"
    )

if not os.path.exists(SATELLITE_IMAGE):
    raise FileNotFoundError(
        f"Satellite image not found: {SATELLITE_IMAGE}"
    )

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# LOAD CNN MODEL
# ============================================================

print("Loading CNN satellite model...")

model = tf.keras.models.load_model(
    MODEL_PATH
)

print("CNN model loaded successfully.")


# ============================================================
# LOAD SATELLITE IMAGE
# ============================================================

print()
print(f"Reading satellite image: {SATELLITE_IMAGE}")

img = image.load_img(
    SATELLITE_IMAGE,
    target_size=(128, 128)
)

img_array = image.img_to_array(
    img
)

img_array = img_array / 255.0

img_array = np.expand_dims(
    img_array,
    axis=0
)


# ============================================================
# CNN PREDICTION
# ============================================================

prediction = model.predict(
    img_array,
    verbose=0
)[0][0]


# Keras class mapping:
# clear = 0
# cloudy = 1

if prediction >= 0.5:

    satellite_class = "cloudy"
    confidence = prediction

else:

    satellite_class = "clear"
    confidence = 1 - prediction


# ============================================================
# REMOTE-SENSING INTERPRETATION
# ============================================================

if satellite_class == "cloudy":

    interpretation = (
        "Satellite imagery indicates cloudy conditions. "
        "Cloud information can support weather monitoring "
        "and decision-support analysis."
    )

else:

    interpretation = (
        "Satellite imagery indicates relatively clear "
        "conditions. The observation can support weather "
        "monitoring and decision-support analysis."
    )


# ============================================================
# INTEGRATION RESULT
# ============================================================

result = {

    "system": (
        "Weather Intelligence AI "
        "Remote-Sensing Integration"
    ),

    "satellite_image": SATELLITE_IMAGE,

    "computer_vision": {

        "model": (
            "CNN Satellite "
            "Clear vs Cloudy Classifier"
        ),

        "predicted_class": satellite_class,

        "confidence_percent": round(
            float(confidence * 100),
            2
        )

    },

    "weather_intelligence": {

        "remote_sensing_source": (
            "Satellite imagery"
        ),

        "weather_variable_supported": (
            "Cloud conditions"
        )

    },

    "interpretation": interpretation
}


# ============================================================
# SAVE RESULT
# ============================================================

with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        result,
        f,
        indent=4
    )


# ============================================================
# DISPLAY RESULT
# ============================================================

print()
print("========================================")
print("REMOTE-SENSING INTEGRATION RESULT")
print("========================================")

print(
    f"Satellite Image : {SATELLITE_IMAGE}"
)

print(
    f"Classification   : {satellite_class}"
)

print(
    f"Confidence       : "
    f"{confidence * 100:.2f}%"
)

print(
    "Weather Variable : Cloud Conditions"
)

print()
print(
    "Remote-sensing integration completed "
    "successfully."
)

print(
    f"Result saved to  : {OUTPUT_FILE}"
)