from ultralytics import YOLO
from pathlib import Path

# -----------------------------
# CONFIGURATION
# -----------------------------
MODEL_PATH = "../weights/best.pt"            # Path to your trained model
INPUT_FOLDER = "../images"                   # Folder with images to predict
OUTPUT_FOLDER = "../predictions"             # Folder to save annotated images
IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png"]  # Supported image formats

# -----------------------------
# LOAD MODEL
# -----------------------------
model = YOLO(MODEL_PATH)

# -----------------------------
# FIND ALL IMAGES
# -----------------------------
input_path = Path(INPUT_FOLDER)
image_files = [f for f in input_path.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS]

if not image_files:
    print(f"No images found in '{INPUT_FOLDER}' with extensions {IMAGE_EXTENSIONS}")
    exit()

# -----------------------------
# ENSURE OUTPUT FOLDER EXISTS
# -----------------------------
Path(OUTPUT_FOLDER).mkdir(parents=True, exist_ok=True)

# -----------------------------
# RUN PREDICTIONS
# -----------------------------
for img_path in image_files:
    print(f"\nProcessing image: {img_path.name}")
    
    # Run prediction and save directly into OUTPUT_FOLDER
    results = model(
        str(img_path),
        save=True,
        project=OUTPUT_FOLDER,  # Saves annotated images here
        exist_ok=True           # Overwrite if necessary
    )
    
    # Access results
    result = results[0]
    if result.boxes is None or len(result.boxes) == 0:
        print("No objects detected.")
        continue
    
    # Print all detections
    for box in result.boxes:
        cls_index = int(box.cls)
        class_name = result.names[cls_index]
        confidence = float(box.conf)
        xyxy = box.xyxy[0].tolist()
        print(f"Class: {class_name}, Index: {cls_index}, Confidence: {confidence:.2f}, Box: {xyxy}")

print("\nAll predictions saved directly to:", OUTPUT_FOLDER)
