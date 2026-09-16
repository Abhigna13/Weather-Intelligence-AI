import os
import numpy as np
import tensorflow as tf

from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image


# ============================================================
# PATHS
# ============================================================

DATA_DIR = "satellite/dataset"
MODEL_DIR = "api/models/deep_learning"
REPORT_DIR = "reports/satellite"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)


# ============================================================
# DATASET CHECK
# ============================================================

classes = ["cloudy", "clear"]

for class_name in classes:

    class_path = os.path.join(DATA_DIR, class_name)

    if not os.path.exists(class_path):
        raise FileNotFoundError(
            f"Dataset folder not found: {class_path}"
        )

    image_count = len([
        f for f in os.listdir(class_path)
        if f.lower().endswith(
            (".jpg", ".jpeg", ".png")
        )
    ])

    print(f"{class_name} images: {image_count}")

    if image_count < 2:
        raise ValueError(
            f"Add at least 2 images to {class_path}"
        )


# ============================================================
# IMAGE SETTINGS
# ============================================================

IMAGE_SIZE = (128, 128)
BATCH_SIZE = 16


# ============================================================
# DATA PREPROCESSING
# ============================================================

datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2
)


train_data = datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="training",
    shuffle=True
)


validation_data = datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="validation",
    shuffle=False
)


print()
print("Class mapping:")
print(train_data.class_indices)


# ============================================================
# CNN MODEL
# ============================================================

model = models.Sequential([

    layers.Input(
        shape=(128, 128, 3)
    ),

    layers.Conv2D(
        32,
        (3, 3),
        activation="relu"
    ),

    layers.MaxPooling2D(
        (2, 2)
    ),

    layers.Conv2D(
        64,
        (3, 3),
        activation="relu"
    ),

    layers.MaxPooling2D(
        (2, 2)
    ),

    layers.Conv2D(
        128,
        (3, 3),
        activation="relu"
    ),

    layers.MaxPooling2D(
        (2, 2)
    ),

    layers.Flatten(),

    layers.Dense(
        64,
        activation="relu"
    ),

    layers.Dropout(0.3),

    layers.Dense(
        1,
        activation="sigmoid"
    )
])


# ============================================================
# COMPILE
# ============================================================

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)


print()
print("CNN model created successfully.")


# ============================================================
# TRAIN
# ============================================================

history = model.fit(
    train_data,
    validation_data=validation_data,
    epochs=10
)


# ============================================================
# EVALUATION
# ============================================================

loss, accuracy = model.evaluate(
    validation_data,
    verbose=0
)

print()
print("CNN Validation Results")
print("----------------------")
print(f"Validation Loss     : {loss:.4f}")
print(f"Validation Accuracy : {accuracy * 100:.2f}%")


# ============================================================
# SAVE MODEL
# ============================================================

model_path = os.path.join(
    MODEL_DIR,
    "cnn_satellite_cloud_model.keras"
)

model.save(model_path)


# ============================================================
# SAVE TRAINING HISTORY
# ============================================================

history_path = os.path.join(
    REPORT_DIR,
    "cnn_training_history.csv"
)

with open(history_path, "w") as f:

    f.write(
        "epoch,accuracy,validation_accuracy,"
        "loss,validation_loss\n"
    )

    for i in range(len(history.history["accuracy"])):

        f.write(
            f"{i + 1},"
            f"{history.history['accuracy'][i]:.6f},"
            f"{history.history['val_accuracy'][i]:.6f},"
            f"{history.history['loss'][i]:.6f},"
            f"{history.history['val_loss'][i]:.6f}\n"
        )


# ============================================================
# SAVE EVALUATION SUMMARY
# ============================================================

evaluation_path = os.path.join(
    REPORT_DIR,
    "cnn_satellite_evaluation.txt"
)

with open(evaluation_path, "w") as f:

    f.write("Satellite CNN Classification Evaluation\n")
    f.write("----------------------------------------\n")
    f.write("Task: Clear vs Cloudy satellite image classification\n")
    f.write("Model: Convolutional Neural Network (CNN)\n")
    f.write(f"Image Size: {IMAGE_SIZE}\n")
    f.write("Epochs: 10\n")
    f.write(f"Validation Loss: {loss:.4f}\n")
    f.write(
        f"Validation Accuracy: {accuracy * 100:.2f}%\n"
    )


# ============================================================
# SAMPLE IMAGE PREDICTION
# ============================================================

sample_image = os.path.join(
    DATA_DIR,
    "clear",
    "clear3.jpg"
)

if os.path.exists(sample_image):

    img = image.load_img(
        sample_image,
        target_size=IMAGE_SIZE
    )

    img_array = image.img_to_array(img)

    img_array = img_array / 255.0

    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    prediction = model.predict(
        img_array,
        verbose=0
    )[0][0]

    # Keras directory ordering:
    # clear = 0
    # cloudy = 1

    if prediction >= 0.5:
        predicted_class = "cloudy"
        confidence = prediction
    else:
        predicted_class = "clear"
        confidence = 1 - prediction

    print()
    print("Sample Satellite Image Prediction")
    print("----------------------------------")
    print(f"Image            : {sample_image}")
    print(f"Predicted Class  : {predicted_class}")
    print(f"Confidence       : {confidence * 100:.2f}%")


# ============================================================
# FINAL OUTPUT
# ============================================================

print()
print("CNN satellite model trained successfully!")
print(f"Model saved to       : {model_path}")
print(f"Evaluation saved to  : {evaluation_path}")
print(f"History saved to     : {history_path}")