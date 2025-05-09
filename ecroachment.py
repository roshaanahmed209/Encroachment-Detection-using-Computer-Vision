import cv2
import numpy as np
import json
import os
import datetime

# Hardcoded paths
USER_IMAGE_DIR = r"H:\fyp\matched\user_in"
SITEMAP_IMAGE_DIR = r"H:\fyp\matched\sitemap"
OUTPUT_DIR = r"H:\fyp\resultant"

def create_individual_masks(json_data, image_shape):
    masks = []
    if "segments" in json_data:
        for segment in json_data["segments"]:
            mask = np.zeros(image_shape[:2], dtype=np.uint8)
            polygon = np.array(segment, dtype=np.int32)
            cv2.fillPoly(mask, [polygon], 255)
            masks.append(mask)
    else:
        raise ValueError("JSON data does not contain 'segments' key.")
    print("Masks created successfully.")
    return masks

def create_combined_mask(masks):
    combined = np.zeros_like(masks[0])
    for mask in masks:
        combined = cv2.bitwise_or(combined, mask)
    print("Combined mask created successfully.")
    return combined

def calculate_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    if np.sum(union) == 0:
        return 0.0
    print("IoU calculated successfully.")
    return np.sum(intersection) / np.sum(union)

def detect_encroachments(google_masks, sitemap_masks, google_json, iou_threshold=0.1):
    encroachment_mask = np.zeros_like(google_masks[0])
    individual_encroachment_segments = []

    for g_mask, g_segment in zip(google_masks, google_json["segments"]):
        matched = any(calculate_iou(g_mask, s_mask) >= iou_threshold for s_mask in sitemap_masks)
        if not matched:
            encroachment_mask = cv2.add(encroachment_mask, g_mask)
            individual_encroachment_segments.append(g_segment)

    print("Encroachment mask created successfully.")
    return encroachment_mask, individual_encroachment_segments

def mark_encroachments_blend(google_image, encroachment_mask):
    overlay = google_image.copy()
    red = np.zeros_like(google_image)
    red[:, :, 2] = 255  # red channel

    # Create the blended image using np.where for the mask area
    mask_bool = encroachment_mask.astype(bool)
    overlay = np.where(mask_bool[:, :, None], 
                       cv2.addWeighted(google_image, 0.5, red, 0.5, 0), 
                       google_image)

    print("Encroachment marked successfully.")
    return overlay

def get_latest_file_by_ext(directory, extensions):
    files = [f for f in os.listdir(directory) if os.path.splitext(f)[1].lower() in extensions]
    if not files:
        raise FileNotFoundError(f"No files with extensions {extensions} found in {directory}")
    files.sort(key=lambda x: os.path.getmtime(os.path.join(directory, x)), reverse=True)
    return os.path.join(directory, files[0])

def process_from_matched_dirs():
    """
    Process encroachment detection using latest user input and sitemap from hardcoded folders.

    Returns:
        Tuple of (result_image_path, result_json_path, num_encroachments)
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Processing encroachment detection...")

    # Get latest files
    user_image_path = get_latest_file_by_ext(USER_IMAGE_DIR, ['.png', '.jpg', '.jpeg'])
    user_json_path = get_latest_file_by_ext(USER_IMAGE_DIR, ['.json'])
    sitemap_image_path = get_latest_file_by_ext(SITEMAP_IMAGE_DIR, ['.png', '.jpg', '.jpeg'])
    sitemap_json_path = get_latest_file_by_ext(SITEMAP_IMAGE_DIR, ['.json'])

    print(f"User image: {user_image_path}")
    print(f"User JSON : {user_json_path}")
    print(f"Sitemap image: {sitemap_image_path}")
    print(f"Sitemap JSON : {sitemap_json_path}")

    # Load images
    google_image = cv2.imread(user_image_path)
    sitemap_image = cv2.imread(sitemap_image_path)
    if google_image is None or sitemap_image is None:
        raise ValueError("Could not load one or both images.")

    # Load JSONs
    with open(user_json_path, "r") as f:
        google_json = json.load(f)
    with open(sitemap_json_path, "r") as f:
        sitemap_json = json.load(f)

    # Create masks
    google_masks = create_individual_masks(google_json, google_image.shape)
    sitemap_masks = create_individual_masks(sitemap_json, sitemap_image.shape)
    sitemap_masks_resized = [cv2.resize(mask, (google_image.shape[1], google_image.shape[0])) for mask in sitemap_masks]

    # Detect encroachments
    encroachment_mask, encroachment_segments = detect_encroachments(google_masks, sitemap_masks_resized, google_json)

    # Create blended overlay
    result_image = mark_encroachments_blend(google_image, encroachment_mask)

    # Save results
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    result_image_path = os.path.join(OUTPUT_DIR, f"encroachment_result_{timestamp}.png")
    result_json_path = os.path.join(OUTPUT_DIR, f"encroachment_mask_{timestamp}.json")
    original_user_image_path = os.path.join(OUTPUT_DIR, f"original_user_{timestamp}.png")

    cv2.imwrite(result_image_path, result_image)
    cv2.imwrite(original_user_image_path, google_image)
    with open(result_json_path, "w") as f:
        json.dump({"segments": encroachment_segments}, f, indent=2)

    print("✅ All outputs saved to:", OUTPUT_DIR)
    return result_image_path, result_json_path, len(encroachment_segments)

# if __name__ == "__main__":
#     process_from_matched_dirs()
