from ultralytics import YOLO
import os
import cv2
import json

# Set paths
image_folder = r"H:\fyp\FYP_new_dt\FYP_new_dt\Sitemap\images"
output_folder = r"H:\fyp\FYP_new_dt\FYP_new_dt\Sitemap\labels"
weights_path = r"best_model_weights_site_map.pt"

# Create output directory if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Load YOLOv8 segmentation model
model = YOLO(weights_path)

# Loop through images and process
for filename in os.listdir(image_folder):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        image_path = os.path.join(image_folder, filename)
        results = model(image_path)

        # Extract mask polygons
        segments = []
        for result in results:
            if result.masks is not None:
                for xy in result.masks.xy:
                    polygon = [[int(x), int(y)] for x, y in xy]
                    segments.append(polygon)

        # Save as JSON
        json_data = {"segments": segments}
        output_json_path = os.path.join(output_folder, os.path.splitext(filename)[0] + ".json")
        with open(output_json_path, "w") as f:
            json.dump(json_data, f, indent=2)

        print(f"Saved segmentation for: {filename}")
