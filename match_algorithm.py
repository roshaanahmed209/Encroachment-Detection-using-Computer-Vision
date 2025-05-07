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

def load_masks_from_directory(directory):
    masks = []
    mask_names = []
    for file in sorted(os.listdir(directory)):
        ext = os.path.splitext(file)[1].lower()
        if ext in VALID_IMAGE_EXTENSIONS:
            path = os.path.join(directory, file)
            mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                masks.append(extract_features(mask))
                mask_names.append(file)
    return masks, mask_names

def compute_similarity_matrix(user_masks, sitemap_dir, labels_dir):
    similarity_scores = []
    sitemap_masks, mask_names = load_masks_from_directory(labels_dir)
    for sitemap_mask in sitemap_masks:
        row = []
        for user_mask in user_masks:
            sim = cosine_similarity([user_mask], [sitemap_mask])[0][0]
            row.append(sim)
        similarity_scores.append(row)
    return similarity_scores

def average_similarity(sim_matrix):
    return np.mean(sim_matrix, axis=1).tolist()

def process_all_sitemaps(user_masks):
    all_sitemap_scores = []
    sitemap_files = sorted(os.listdir(ANNOTATED_SEGMENTATIONS_DIR))

    seen_bases = set()
    for file in sitemap_files:
        base_name, ext = os.path.splitext(file)
        if ext.lower() not in VALID_IMAGE_EXTENSIONS or base_name in seen_bases:
            continue
        seen_bases.add(base_name)

        matching_masks = [f for f in sitemap_files if f.startswith(base_name)]
        labels_dir = ANNOTATED_SEGMENTATIONS_DIR
        img_dir = ANNOTATED_SITEMAPS_DIR

        sitemap_mask_paths = [os.path.join(labels_dir, f) for f in matching_masks]
        sitemap_masks = []
        for path in sitemap_mask_paths:
            mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                sitemap_masks.append(extract_features(mask))

        if not sitemap_masks:
            continue

        sim_matrix = []
        for s_mask in sitemap_masks:
            row = []
            for u_mask in user_masks:
                sim = cosine_similarity([u_mask], [s_mask])[0][0]
                row.append(sim)
            sim_matrix.append(row)

        avg_sim = np.mean(sim_matrix)
        all_sitemap_scores.append((base_name, avg_sim))

    return all_sitemap_scores

def save_match_and_user_data(best_base_name, user_image_path, user_mask_paths):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for ext in VALID_IMAGE_EXTENSIONS:
        img_path = os.path.join(ANNOTATED_SITEMAPS_DIR, best_base_name + ext)
        if os.path.exists(img_path):
            shutil.copy2(img_path, os.path.join(SITEMAP_OUT_DIR, f"matched_{timestamp}_{os.path.basename(img_path)}"))
            break

    matching_masks = [f for f in os.listdir(ANNOTATED_SEGMENTATIONS_DIR) if f.startswith(best_base_name)]
    for mask_file in matching_masks:
        src = os.path.join(ANNOTATED_SEGMENTATIONS_DIR, mask_file)
        dst = os.path.join(SITEMAP_OUT_DIR, f"matched_{timestamp}_{mask_file}")
        shutil.copy2(src, dst)

    shutil.copy2(user_image_path, os.path.join(USER_IN_OUT_DIR, f"user_{timestamp}_{os.path.basename(user_image_path)}"))
    for mask_path in user_mask_paths:
        shutil.copy2(mask_path, os.path.join(USER_IN_OUT_DIR, f"user_{timestamp}_{os.path.basename(mask_path)}"))

def json_to_mask(json_path, image_shape):
    with open(json_path, 'r') as f:
        data = json.load(f)
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    for seg in data.get('segments', []):
        pts = np.array(seg, dtype=np.int32)
        cv2.fillPoly(mask, [pts], 255)
    return mask

def main(user_image_path=None, user_json_path=None):
    if user_image_path and user_json_path:
        image = cv2.imread(user_image_path)
        if image is None:
            print(f"Cannot read image: {user_image_path}")
            return None

        with open(user_json_path) as f:
            data = json.load(f)

        user_masks = []
        temp_paths = []

        for i, segment in enumerate(data.get('segments', [])):
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            pts = np.array(segment, dtype=np.int32)
            cv2.fillPoly(mask, [pts], 255)
            path = os.path.join(os.path.dirname(user_image_path), f"temp_mask_{i}.png")
            cv2.imwrite(path, mask)
            user_masks.append(extract_features(mask))
            temp_paths.append(path)

        scores = process_all_sitemaps(user_masks)
        if not scores:
            print("No sitemaps to compare.")
            return None

        scores.sort(key=lambda x: -x[1])  # Highest cosine similarity = most similar
        best_match = scores[0]
        print(f"Best match: {best_match[0]}, Similarity: {best_match[1]:.4f}")

        save_match_and_user_data(best_match[0], user_image_path, temp_paths)

        for path in temp_paths:
            os.remove(path)

        return best_match[0]

# Example usage:
# main(r"path_to_user_image.jpg", r"path_to_user_segmentation.json")
