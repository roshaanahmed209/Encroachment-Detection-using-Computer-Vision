import cv2
import numpy as np
import json
import os

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
    return masks

def create_combined_mask(masks):
    combined = np.zeros_like(masks[0])
    for mask in masks:
        combined = cv2.bitwise_or(combined, mask)
    return combined

def calculate_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    if np.sum(union) == 0:
        return 0.0
    return np.sum(intersection) / np.sum(union)

def detect_encroachments(google_masks, sitemap_masks, google_json, iou_threshold=0.1):
    encroachment_mask = np.zeros_like(google_masks[0])
    individual_encroachment_segments = []

    for g_mask, g_segment in zip(google_masks, google_json["segments"]):
        matched = any(calculate_iou(g_mask, s_mask) >= iou_threshold for s_mask in sitemap_masks)
        if not matched:
            encroachment_mask = cv2.add(encroachment_mask, g_mask)
            individual_encroachment_segments.append(g_segment)  # Save unmatched segment

    return encroachment_mask, individual_encroachment_segments

def mark_encroachments_blend(google_image, encroachment_mask):
    overlay = google_image.copy()
    red = np.zeros_like(google_image)
    red[:, :, 2] = 255
    mask_bool = encroachment_mask.astype(bool)
    overlay[mask_bool] = cv2.addWeighted(google_image[mask_bool], 0.5, red[mask_bool], 0.5, 0)
    return overlay

def process_and_save(google_image_path, google_json_path,
                     sitemap_image_path, sitemap_json_path,
                     output_folder="resultant"):
    os.makedirs(output_folder, exist_ok=True)

    # Load images
    google_image = cv2.imread(google_image_path)
    sitemap_image = cv2.imread(sitemap_image_path)
    if google_image is None or sitemap_image is None:
        raise ValueError("Could not load one or both images.")

    # Load JSONs
    with open(google_json_path, "r") as f:
        google_json = json.load(f)
    with open(sitemap_json_path, "r") as f:
        sitemap_json = json.load(f)

    # Create masks
    google_masks = create_individual_masks(google_json, google_image.shape)
    sitemap_masks = create_individual_masks(sitemap_json, sitemap_image.shape)
    sitemap_masks_resized = [cv2.resize(mask, (google_image.shape[1], google_image.shape[0])) for mask in sitemap_masks]

    # Detect encroachments
    encroachment_mask, encroachment_segments = detect_encroachments(google_masks, sitemap_masks_resized, google_json)

    # Overlay
    result_image = mark_encroachments_blend(google_image, encroachment_mask)

    # Save image and JSON
    result_image_path = os.path.join(output_folder, "encroachment_result.png")
    result_json_path = os.path.join(output_folder, "encroachment_mask.json")
    cv2.imwrite(result_image_path, result_image)
    with open(result_json_path, "w") as f:
        json.dump({"segments": encroachment_segments}, f, indent=2)

    # Return useful info for the web
    return result_image_path, result_json_path, len(encroachment_segments)
