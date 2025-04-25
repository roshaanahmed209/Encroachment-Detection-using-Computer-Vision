import cv2
import numpy as np
import json

def create_mask_from_json(json_data, image_shape):
    """
    Creates a segmentation mask from JSON data containing segmentation values.
    """
    mask = np.zeros(image_shape[:2], dtype=np.uint8)  # Initialize an empty mask
    if "segments" in json_data:
        for segment in json_data["segments"]:
            polygon = np.array(segment, dtype=np.int32)
            cv2.fillPoly(mask, [polygon], 255)  # Fill the polygons on the mask
    else:
        raise ValueError("JSON data does not contain 'segments' key.")

    return mask

def calculate_iou(mask1, mask2):
    """
    Calculates the Intersection over Union (IoU) between two binary masks.
    """
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    iou = np.sum(intersection) / np.sum(union)
    return iou

def detect_encroachments(uploaded_mask, sitemap_mask, iou_threshold=0.1):
    """
    Detects encroachments by comparing the uploaded mask with the sitemap mask.
    Returns a mask highlighting encroachments.
    """
    encroachment_mask = np.zeros_like(uploaded_mask)

    # Find contours in the uploaded mask
    contours, _ = cv2.findContours(uploaded_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        # Create a mask for the current building
        building_mask = np.zeros_like(uploaded_mask)
        cv2.drawContours(building_mask, [contour], -1, 255, thickness=cv2.FILLED)

        # Calculate IoU with the sitemap mask
        iou = calculate_iou(building_mask, sitemap_mask)

        # If IoU is below the threshold, mark as encroachment
        if iou < iou_threshold:
            encroachment_mask = cv2.add(encroachment_mask, building_mask)

    return encroachment_mask

def mark_encroachments(image, encroachment_mask):
    """
    Marks encroachments on the image with a see-through red color.
    """
    # Create a red overlay
    red_overlay = np.zeros_like(image)
    red_overlay[:, :, 2] = 255  # Set red channel to maximum

    # Blend the red overlay with the image
    blended_image = cv2.addWeighted(image, 0.7, red_overlay, 0.3, 0)

    # Apply the encroachment mask
    blended_image[encroachment_mask == 255] = red_overlay[encroachment_mask == 255]

    return blended_image

def process_encroachment_detection(uploaded_image_path, sitemap_image_path, json_path):
    """
    Processes the uploaded image and the closest matching sitemap to detect encroachments.
    """
    # Load the uploaded image
    uploaded_image = cv2.imread(uploaded_image_path)
    if uploaded_image is None:
        raise ValueError("Failed to load the uploaded image.")

    # Load the closest matching sitemap image
    sitemap_image = cv2.imread(sitemap_image_path, cv2.IMREAD_GRAYSCALE)
    if sitemap_image is None:
        raise ValueError("Failed to load the sitemap image.")

    # Load segmentation JSON
    with open(json_path, "r") as json_file:
        json_data = json.load(json_file)

    # Create segmentation mask from JSON data
    uploaded_mask = create_mask_from_json(json_data, uploaded_image.shape)

    # Resize the sitemap mask to match the uploaded image dimensions
    sitemap_mask = cv2.resize(sitemap_image, (uploaded_mask.shape[1], uploaded_mask.shape[0]))

    # Detect encroachments
    encroachment_mask = detect_encroachments(uploaded_mask, sitemap_mask)

    # Mark encroachments on the uploaded image
    result_image = mark_encroachments(uploaded_image, encroachment_mask)

    return result_image

# Example usage
if __name__ == "__main__":
    # Paths to the uploaded image, closest matching sitemap, and segmentation JSON
    uploaded_image_path = "uploaded_image.jpg"
    sitemap_image_path = "closest_match_sitemap.jpg"
    json_path = "segmentation_results.json"

    try:
        # Process the images to detect encroachments
        result_image = process_encroachment_detection(uploaded_image_path, sitemap_image_path, json_path)

        # Display the result
        cv2.imshow("Encroachment Detection", result_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Save the result
        cv2.imwrite("encroachment_result.jpg", result_image)
        print("Encroachment detection completed. Result saved as 'encroachment_result.jpg'.")
    except Exception as e:
        print(f"Error: {e}")