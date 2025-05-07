import cv2
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os
import shutil
from datetime import datetime
import re
import json

VALID_IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}

ANNOTATED_SITEMAPS_DIR = r"H:\fyp\FYP_new_dt\FYP_new_dt\Sitemap\images"
ANNOTATED_SEGMENTATIONS_DIR = r"H:\fyp\FYP_new_dt\FYP_new_dt\Sitemap\labels"
USER_INPUT_DIR = r"H:\fyp\from_web_1"

MATCHED_DIR = r"H:\fyp\matched"
SITEMAP_OUT_DIR = os.path.join(MATCHED_DIR, "sitemap")
USER_IN_OUT_DIR = os.path.join(MATCHED_DIR, "user_in")

os.makedirs(SITEMAP_OUT_DIR, exist_ok=True)
os.makedirs(USER_IN_OUT_DIR, exist_ok=True)

def extract_features(mask):
    resized = cv2.resize(mask, (128, 128))
    flat = resized.flatten().astype(np.float32)
    norm = np.linalg.norm(flat)
    return flat / norm if norm > 0 else flat

def load_sitemap_masks(sitemap_base_name):
    masks = []
    for ext in VALID_IMAGE_EXTENSIONS:
        mask_path = os.path.join(ANNOTATED_SEGMENTATIONS_DIR, f"{sitemap_base_name}{ext}")
        if os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                masks.append(extract_features(mask))
    return masks

def load_user_masks_from_json(json_path, image_shape):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    masks = []
    for segment in data.get('segments', []):
        mask = np.zeros(image_shape[:2], dtype=np.uint8)
        points = np.array(segment, dtype=np.int32)
        cv2.fillPoly(mask, [points], 255)
        masks.append(extract_features(mask))
    return masks

def compute_similarity_matrix(user_masks, sitemap_masks):
    similarity_matrix = []
    for sitemap_mask in sitemap_masks:
        row = []
        for user_mask in user_masks:
            # Convert similarity to distance (1 - similarity)
            similarity = cosine_similarity([user_mask], [sitemap_mask])[0][0]
            distance = 1 - similarity
            row.append(distance)
        similarity_matrix.append(row)
    return similarity_matrix

def find_best_match(user_image_path, user_json_path):
    # Load user image and masks
    image = cv2.imread(user_image_path)
    if image is None:
        raise ValueError(f"Cannot read user image: {user_image_path}")
    
    user_masks = load_user_masks_from_json(user_json_path, image.shape)
    if not user_masks:
        raise ValueError("No valid masks found in user JSON")

    # Get all sitemap base names
    sitemap_files = os.listdir(ANNOTATED_SEGMENTATIONS_DIR)
    sitemap_bases = set()
    for file in sitemap_files:
        base_name = os.path.splitext(file)[0]
        sitemap_bases.add(base_name)

    # Store results for each sitemap
    sitemap_results = []
    
    for base_name in sitemap_bases:
        sitemap_masks = load_sitemap_masks(base_name)
        if not sitemap_masks:
            continue

        # Compute similarity matrix for this sitemap
        similarity_matrix = compute_similarity_matrix(user_masks, sitemap_masks)
        
        # Calculate average distance for this sitemap
        avg_distance = np.mean(similarity_matrix)
        sitemap_results.append((base_name, avg_distance, similarity_matrix))

    if not sitemap_results:
        raise ValueError("No valid sitemaps found for comparison")

    # Find sitemap with lowest average distance
    best_match = min(sitemap_results, key=lambda x: x[1])
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save matched sitemap image and masks
    for ext in VALID_IMAGE_EXTENSIONS:
        sitemap_img_path = os.path.join(ANNOTATED_SITEMAPS_DIR, f"{best_match[0]}{ext}")
        if os.path.exists(sitemap_img_path):
            shutil.copy2(sitemap_img_path, 
                        os.path.join(SITEMAP_OUT_DIR, f"matched_{timestamp}_{os.path.basename(sitemap_img_path)}"))
            break

    # Save user input
    shutil.copy2(user_image_path, 
                os.path.join(USER_IN_OUT_DIR, f"user_{timestamp}_{os.path.basename(user_image_path)}"))
    shutil.copy2(user_json_path, 
                os.path.join(USER_IN_OUT_DIR, f"user_{timestamp}_{os.path.basename(user_json_path)}"))

    return {
        'best_match': best_match[0],
        'average_distance': best_match[1],
        'similarity_matrix': best_match[2],
        'timestamp': timestamp
    }

def main(user_image_path=None, user_json_path=None):
    if not user_image_path or not user_json_path:
        print("Please provide both user image and JSON paths")
        return None

    try:
        result = find_best_match(user_image_path, user_json_path)
        print(f"Best match: {result['best_match']}")
        print(f"Average distance: {result['average_distance']:.4f}")
        return result['best_match']
    except Exception as e:
        print(f"Error: {str(e)}")
        return None

# Example usage:
# main(r"path_to_user_image.jpg", r"path_to_user_segmentation.json")
