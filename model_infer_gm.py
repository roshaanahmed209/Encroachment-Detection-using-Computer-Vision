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
    Processes a Google Map image, saves the segmentation results as a JSON file, and returns the file path.
    """
    print("Image uploaded and saved successfully.")

    try:
        # Perform segmentation
        annotated_image, segmentation_mask, segments = segment_google_map(image_path)

        # Convert the annotated image to a format suitable for return
        _, annotated_image_encoded = cv2.imencode('.jpg', annotated_image)
        annotated_image_b64 = base64.b64encode(annotated_image_encoded).decode('utf-8')

        # Convert the segmentation mask to a PIL image
        mask_pil = Image.fromarray(segmentation_mask)

        # Convert the PIL image to a base64-encoded string
        mask_io = io.BytesIO()
        mask_pil.save(mask_io, format='PNG')
        mask_io.seek(0)
        mask_b64 = base64.b64encode(mask_io.getvalue()).decode('utf-8')

        # Create results dictionary
        results = {
            "message": "Segmentation performed successfully",
            "segmentation_mask_b64": mask_b64,
            "annotated_image_b64": annotated_image_b64,
            "segments": segments,  # Include polygon data
            "success": True
        }

        # Define the directory to save JSON files
        save_dir = "segmentation_results"
        os.makedirs(save_dir, exist_ok=True)

        # Create a unique filename using timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_filename = f"segmentation_results_{timestamp}.json"
        json_file_path = os.path.join(save_dir, json_filename)

        # Save the results as a JSON file
        with open(json_file_path, "w") as json_file:
            json.dump(results, json_file, indent=4)

        # Return the path to the JSON file
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
