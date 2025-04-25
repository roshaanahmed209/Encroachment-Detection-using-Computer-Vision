import subprocess
import os
import random
import shutil
from pathlib import Path
from ultralytics import YOLO
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# Step 1: Download the dataset zip file using curl
subprocess.run([
    "curl", "-L",
    "https://app.roboflow.com/ds/nPe7sfQ23a?key=36NX0o7zl6",
    "-o", "roboflow.zip"
])

# Step 2: Unzip the downloaded file
subprocess.run(["unzip", "roboflow.zip", "-d", "roboflow_dataset"])

# Step 3: Remove the zip file
if os.path.exists("roboflow.zip"):
    os.remove("roboflow.zip")
else:
    print("roboflow.zip does not exist.")

# Step 4: Install ultralytics package
subprocess.run(["pip", "install", "ultralytics"])

# Step 5: Initialize YOLO model
model = YOLO("yolov8m-seg.pt")

# Step 6: Write data.yaml file for training
data_yaml_path = "data.yaml"
with open(data_yaml_path, "w") as f:
    f.write(
        """
train: roboflow_dataset/train/images
val: roboflow_dataset/valid/images

nc: 1
names: ['Buildings']

roboflow:
  workspace: encroachment
  project: encroachment-dwic8
  version: 24
  license: CC BY 4.0
  url: https://universe.roboflow.com/encroachment/encroachment-dwic8/dataset/24
        """
    )

# Step 7: Split dataset into training and validation sets
train_images_dir = "H:/fyp/roboflow_dataset/train/images"
train_labels_dir = "H:/fyp/roboflow_dataset/train/labels"
val_images_dir = "H:/fyp/roboflow_dataset/valid/images"
val_labels_dir = "H:/fyp/roboflow_dataset/valid/labels"


os.makedirs(val_images_dir, exist_ok=True)
os.makedirs(val_labels_dir, exist_ok=True)

image_files = [f for f in os.listdir(train_images_dir) if os.path.isfile(os.path.join(train_images_dir, f))]
random.shuffle(image_files)

split_idx = int(0.8 * len(image_files))
val_images = image_files[split_idx:]

for img in val_images:
    src_image_path = os.path.join(train_images_dir, img)
    dest_image_path = os.path.join(val_images_dir, img)
    shutil.move(src_image_path, dest_image_path)

    label_file = Path(train_labels_dir) / (Path(img).stem + ".txt")
    if label_file.exists():
        dest_label_path = os.path.join(val_labels_dir, label_file.name)
        shutil.move(label_file, dest_label_path)

print("Dataset splitting into train/valid directories is complete!")

# Step 8: Train the model
model.train(
    data="data.yaml",
    project="output_model",
    name="epoc",
    epochs=150,
    batch=8,
    imgsz=512,
)

# Step 9: Save the trained model for web integration
trained_model_path = "output_model/epoc/weights/best.pt"

# Export the trained model to ONNX format
export_dir = "exported_model"
os.makedirs(export_dir, exist_ok=True)

model = YOLO(trained_model_path)
export_result = model.export(format="onnx", imgsz=512)  # Save as ONNX format
onnx_path = export_result["model"]
print(f"Model exported to: {onnx_path}")

# Step 10: Perform inference on multiple images
samples_dir = "samples"  # Update this with the folder containing your test images
output_dir = "inference_results"
os.makedirs(output_dir, exist_ok=True)

for img_filename in os.listdir(samples_dir):
    if img_filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        sample_image_path = os.path.join(samples_dir, img_filename)
        results = model(sample_image_path)
        result = results[0]

        annotated_image_np = result.plot()
        annotated_image = Image.fromarray(np.uint8(annotated_image_np))
        annotated_image_path = os.path.join(output_dir, f"annotated_{img_filename}")
        annotated_image.save(annotated_image_path)

        plt.imshow(annotated_image)
        plt.axis("off")
        plt.title(f"Model Predictions on {img_filename}")
        plt.show()

        print(f"Predictions for {img_filename}:")
        for i in range(len(result.boxes)):
            label = result.boxes.cls[i].item()
            confidence = result.boxes.conf[i].item()
            coordinates = result.boxes.xyxy[i].tolist()
            print(f"Label: {label}, Confidence: {confidence:.4f}, Coordinates: {coordinates}")
        print("=" * 50)
