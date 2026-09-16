import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# --------------------------------------------------
# Paths
# --------------------------------------------------

DATA_DIR = "satellite/dataset"

MODEL_PATH = (
    "api/models/deep_learning/"
    "cnn_satellite_cloud_model.keras"
)

OUTPUT_DIR = "reports/satellite"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# --------------------------------------------------
# Check model
# --------------------------------------------------

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"CNN model not found: {MODEL_PATH}"
    )


# --------------------------------------------------
# Load trained model
# --------------------------------------------------

model = tf.keras.models.load_model(MODEL_PATH)

print("CNN satellite model loaded successfully!")


# --------------------------------------------------
# Prepare validation data
# --------------------------------------------------

datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2
)

validation_data = datagen.flow_from_directory(
    DATA_DIR,
    target_size=(128, 128),
    batch_size=16,
    class_mode="binary",
    subset="validation",
    shuffle=False
)

print("Validation images:", validation_data.samples)
print("Class mapping:", validation_data.class_indices)


# --------------------------------------------------
# Evaluate model
# --------------------------------------------------

loss, accuracy = model.evaluate(
    validation_data,
    verbose=0
)

print()
print("CNN Satellite Model Performance")
print("--------------------------------")
print(f"Loss     : {loss:.4f}")
print(f"Accuracy : {accuracy:.4f}")
print(f"Accuracy : {accuracy * 100:.2f}%")


# --------------------------------------------------
# Predictions
# --------------------------------------------------

validation_data.reset()

probabilities = model.predict(
    validation_data,
    verbose=0
).flatten()

predicted = (
    probabilities >= 0.5
).astype(int)

actual = validation_data.classes


# --------------------------------------------------
# Classification report
# --------------------------------------------------

class_names = list(
    validation_data.class_indices.keys()
)

print()
print("Classification Report")
print("---------------------")

print(
    classification_report(
        actual,
        predicted,
        target_names=class_names,
        zero_division=0
    )
)


# --------------------------------------------------
# Confusion matrix
# --------------------------------------------------

cm = confusion_matrix(
    actual,
    predicted
)

print("Confusion Matrix")
print(cm)


display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

fig, ax = plt.subplots(
    figsize=(7, 7)
)

display.plot(
    ax=ax,
    values_format="d"
)

plt.title(
    "CNN Satellite Image Classification"
)

plt.tight_layout()

cm_path = os.path.join(
    OUTPUT_DIR,
    "cnn_satellite_confusion_matrix.png"
)

plt.savefig(
    cm_path,
    dpi=300
)

plt.close()


# --------------------------------------------------
# Save accuracy chart
# --------------------------------------------------

plt.figure(figsize=(7, 5))

plt.bar(
    ["CNN Accuracy"],
    [accuracy * 100]
)

plt.ylabel("Accuracy (%)")
plt.title("CNN Satellite Classification Accuracy")

plt.ylim(0, 100)
plt.tight_layout()

accuracy_path = os.path.join(
    OUTPUT_DIR,
    "cnn_satellite_accuracy.png"
)

plt.savefig(
    accuracy_path,
    dpi=300
)

plt.close()


# --------------------------------------------------
# Final output
# --------------------------------------------------

print()
print("CNN satellite evaluation completed successfully!")

print(
    f"Confusion matrix saved to: {cm_path}"
)

print(
    f"Accuracy graph saved to: {accuracy_path}"
)