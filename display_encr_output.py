#display_encr_output.py

import cv2
import numpy as np
import json
import matplotlib.pyplot as plt
import datetime as datetime


def draw_json_mask_overlay(image, mask_json, alpha=0.4, color=(0, 0, 255)):
    """
    Draws a translucent red mask from JSON polygon data on the image.
    """
    overlay = image.copy()
    if "segments" not in mask_json:
        raise ValueError("JSON does not contain 'segments' key.")
    
    for segment in mask_json["segments"]:
        polygon = np.array(segment, dtype=np.int32)
        cv2.fillPoly(overlay, [polygon], color)

    # Blend original image with red overlay
    blended = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
    return blended

# -------- Main Display Block --------
if __name__ == "__main__":
    IMAGE_PATH = "resultant/encroachment_result.png"
    MASK_JSON_PATH = "resultant/encroachment_mask.json"

    try:
        # Load image
        image = cv2.imread(IMAGE_PATH)
        if image is None:
            raise FileNotFoundError(f"Image not found at {IMAGE_PATH}")

        # Load mask JSON
        with open(MASK_JSON_PATH, "r") as f:
            mask_json = json.load(f)

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
