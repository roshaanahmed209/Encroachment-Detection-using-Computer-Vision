import cv2
import numpy as np
from ultralytics import YOLO
import os
import base64
from PIL import Image
import io
import json
from datetime import datetime

# Constants
MODEL_PATH = "best_model_weights.pt"  # Path to the YOLO model

# Load YOLO segmentation model
model = YOLO(MODEL_PATH)

def segment_google_map(image_path):
    """
    Performs segmentation on the uploaded Google Map image.
    Returns the annotated image, segmentation mask, and polygon data.
    """
    results = model(image_path)
    result = results[0]

    # Annotated image
    annotated_image = result.plot()

    # Segmentation mask (binary mask with the detected object areas)
    mask = np.zeros_like(annotated_image[:, :, 0])
    segments = []  # List to store polygon data
    if result.masks is not None:
        for box in result.masks.xy:
            polygon = np.array(box, dtype=np.int32)
            cv2.fillPoly(mask, [polygon], 255)
            segments.append(box.tolist())  # Store polygon data as a list of points

    return annotated_image, mask, segments

def process_image(image_path):
    """
    Processes a Google Map image, saves the segmentation results as a JSON file and annotated image,
    and returns the JSON file path.
    """
    print("Processing image:", image_path)

    try:
        # Perform segmentation
        annotated_image, segmentation_mask, segments = segment_google_map(image_path)

        # Prepare save directory
        save_dir = "from_web_1"
        os.makedirs(save_dir, exist_ok=True)

        # Create a unique filename using timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"user_input_{timestamp}"

        # Save annotated image
        annotated_image_path = os.path.join(save_dir, f"{base_filename}_annotated.jpg")
        cv2.imwrite(annotated_image_path, annotated_image)

        # Save segmentation mask as PNG (optional, if you want the mask image)
        mask_path = os.path.join(save_dir, f"{base_filename}_mask.png")
        cv2.imwrite(mask_path, segmentation_mask)

        # Save segmentation JSON
        json_filename = f"{base_filename}_segmentation.json"
        json_file_path = os.path.join(save_dir, json_filename)
        results = {
            "segments": segments
        }
        with open(json_file_path, "w") as json_file:
            json.dump(results, json_file, indent=4)

        # Return only the JSON file path as expected by backend_home.py
        print(f"Segmentation results saved to {json_file_path}")
        return json_file_path

    except Exception as e:
        raise RuntimeError(f"Failed to process image: {e}")

def create_mask_from_json(json_data, image_shape):
    """
    Creates a mask from JSON data.
    """
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    if "segments" in json_data:
        for polygon in json_data["segments"]:
            points = np.array(polygon, dtype=np.int32)
            cv2.fillPoly(mask, [points], 255)
    else:
        raise ValueError("JSON data does not contain 'segments' key.")
    

    print("Mask created successfully.")
    return mask

# if __name__ == "__main__":
#     # Example usage
#     image_path = "H:/fyp/uploaded_image.jpg"  # Replace with the path to your image
#     try:
#         # Process the image and get the path to the JSON file
#         json_file_path = process_image(image_path)
#         print(f"Segmentation results saved at: {json_file_path}")

#         # Load the JSON data
#         with open(json_file_path, "r") as file:
#             json_data = json.load(file)

#         # Reconstruct the mask from the JSON data
#         image_shape = (720, 1280)  # Replace with the actual image dimensions
#         mask = create_mask_from_json(json_data, image_shape)
#         print("Mask created successfully.")

#     except Exception as e:
#         print(f"Error: {e}")
