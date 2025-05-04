import cv2
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os
import json

# Update to new folders created by YOLOv8
ANNOTATED_SITEMAPS_DIR = r"H:\fyp\FYP_new_dt\FYP_new_dt\Sitemap\images"
ANNOTATED_SEGMENTATIONS_DIR = r"H:\fyp\FYP_new_dt\FYP_new_dt\Sitemap\labels"

def create_mask_from_json(json_data, image_shape):
    """
    Creates a segmentation mask from JSON data containing segmentation values.
    """
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    if "segments" in json_data:
        for segment in json_data["segments"]:
            polygon = np.array(segment, dtype=np.int32)
            cv2.fillPoly(mask, [polygon], 255)
    else:
        raise ValueError("JSON data does not contain 'segments' key.")
    return mask

def extract_features(mask):
    """
    Extracts features from the segmentation mask for comparison.
    """
    resized_mask = cv2.resize(mask, (128, 128))  # Resize to fixed size
    flattened = resized_mask.flatten()
    return flattened / np.linalg.norm(flattened)  # Normalize

def find_matching_sitemap(segmented_features, annotated_sitemaps_dir, segmentation_dir):
    """
    Matches features against local annotated sitemaps.
    """
    max_similarity = 0
    best_match = None

    for file in os.listdir(annotated_sitemaps_dir):
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(annotated_sitemaps_dir, file)
            seg_path = os.path.join(segmentation_dir, os.path.splitext(file)[0] + ".json")

            if not os.path.exists(seg_path):
                continue  # Skip if no corresponding segmentation

            with open(seg_path, "r") as f:
                json_data = json.load(f)

            dummy_img = cv2.imread(img_path)
            if dummy_img is None:
                continue

            mask = create_mask_from_json(json_data, dummy_img.shape)
            features = extract_features(mask)

            similarity = cosine_similarity([segmented_features], [features])[0][0]
            if similarity > max_similarity:
                max_similarity = similarity
                best_match = img_path

    return best_match, max_similarity

def process_image_and_json(image_path, json_path):
    """
    Given an image and segmentation JSON, find the closest matching sitemap.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image at {image_path}")

    with open(json_path, "r") as f:
        json_data = json.load(f)

    seg_mask = create_mask_from_json(json_data, image.shape)
    features = extract_features(seg_mask)

    match_path, sim = find_matching_sitemap(features, ANNOTATED_SITEMAPS_DIR, ANNOTATED_SEGMENTATIONS_DIR)

    if match_path:
        print(f"Matching sitemap: {match_path} (Similarity: {sim:.4f})")
    else:
        print("No matching sitemap found.")

    return os.path.basename(match_path) if match_path else None

# Entry point
def main(image_path, json_path):
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return None
    if not os.path.exists(json_path):
        print(f"JSON not found: {json_path}")
        return None
    return process_image_and_json(image_path, json_path)

# Example usage
# matched_file = main(
#     r"H:\fyp\some_input\input_image.png",
#     r"H:\fyp\some_input\input_image.json"
# )
# print("Matched file:", matched_file)
