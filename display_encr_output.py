import cv2
import numpy as np
import json
import matplotlib.pyplot as plt
import os

def draw_json_mask_overlay(image, mask_json, alpha=0.4, color=(0, 0, 255)):
    """
    Draws a translucent red mask from JSON polygon data on the image.
    """
    overlay = image.copy()
    if "segments" not in mask_json:
        raise ValueError("JSON does not contain 'segments' key.")
    
    for segment in mask_json["segments"]:
        # Ensure the segment has at least 3 points (a valid polygon)
        if len(segment) >= 3:
            polygon = np.array(segment, dtype=np.int32)
            cv2.fillPoly(overlay, [polygon], color)
        else:
            print(f"Skipping invalid segment: {segment}")
    
    # Blend original image with red overlay
    blended = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
    return blended

def get_latest_file_by_ext(directory, extensions):
    """
    Fetch the latest file with the given extensions from the directory.
    """
    files = [f for f in os.listdir(directory) if os.path.splitext(f)[1].lower() in extensions]
    if not files:
        raise FileNotFoundError(f"No files with extensions {extensions} found in {directory}")
    files.sort(key=lambda x: os.path.getmtime(os.path.join(directory, x)), reverse=True)
    return os.path.join(directory, files[0])

def display_encroachment_overlay():
    """
    Main function to fetch the latest image and JSON, apply the overlay, and display the result.
    """
    OUTPUT_DIR = "resultant"
    
    try:
        # Get the latest processed image and JSON from the resultant folder
        image_path = get_latest_file_by_ext(OUTPUT_DIR, ['.png', '.jpg', '.jpeg'])
        mask_json_path = get_latest_file_by_ext(OUTPUT_DIR, ['.json'])

        print(f"Displaying results from {image_path} and {mask_json_path}")

        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image not found at {image_path}")

        # Load mask JSON
        with open(mask_json_path, "r") as f:
            mask_json = json.load(f)

        # Log loaded details
        print(f"Loaded image shape: {image.shape}")
        print(f"Mask JSON segments: {len(mask_json['segments'])}")
        
        # Draw overlay
        result_overlay = draw_json_mask_overlay(image, mask_json)

        # Display
        plt.figure(figsize=(10, 10))
        plt.imshow(cv2.cvtColor(result_overlay, cv2.COLOR_BGR2RGB))
        plt.title("Encroachment Overlay")
        plt.axis("off")
        plt.show()

    except Exception as e:
        print(f"Error displaying output: {e}")

# # Run the function if executed directly
# if __name__ == "__main__":
#     display_encroachment_overlay()
