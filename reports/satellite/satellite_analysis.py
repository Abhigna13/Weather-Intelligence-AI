from pathlib import Path
import cv2
import numpy as np

INPUT_DIR = Path("satellite/images")
OUTPUT_DIR = Path("reports/satellite")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Find satellite images
image_files = [
    p for p in INPUT_DIR.iterdir()
    if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".tif", ".tiff"]
    and "real" in p.stem.lower()
]

if not image_files:
    raise FileNotFoundError(
        "No converted satellite image found in satellite/images"
    )

image_path = image_files[0]

print(f"Reading image: {image_path}")

image = cv2.imread(str(image_path))

if image is None:
    raise ValueError(f"Unable to read image: {image_path}")

height, width, channels = image.shape

print("Image Analysis")
print("----------------------------")
print(f"Image Name : {image_path.name}")
print(f"Width      : {width} pixels")
print(f"Height     : {height} pixels")
print(f"Channels   : {channels}")

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

print(f"Mean Intensity : {np.mean(gray):.2f}")
print(f"Std Intensity  : {np.std(gray):.2f}")

# Edge detection
edges = cv2.Canny(gray, 100, 200)

# Save outputs
gray_path = OUTPUT_DIR / "satellite_grayscale.png"
edges_path = OUTPUT_DIR / "satellite_edges.png"

cv2.imwrite(str(gray_path), gray)
cv2.imwrite(str(edges_path), edges)

print("----------------------------")
print("Satellite analysis completed successfully.")
print(f"Grayscale image : {gray_path}")
print(f"Edge image      : {edges_path}")