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
    resized_mask = cv2.resize(mask, (128, 128))
    flattened = resized_mask.flatten()
    return flattened / np.linalg.norm(flattened)

def find_matching_sitemap(segmented_features, annotated_images_dir, annotated_labels_dir):
    max_similarity = -1
    best_img_path = None
    best_mask_path = None

    for mask_file in os.listdir(annotated_labels_dir):
        ext = os.path.splitext(mask_file)[1].lower()
        if ext not in VALID_IMAGE_EXTENSIONS:
            continue

        mask_path = os.path.join(annotated_labels_dir, mask_file)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue

        features = extract_features(mask)
        similarity = cosine_similarity([segmented_features], [features])[0][0]

        if similarity > max_similarity:
            base_name = os.path.splitext(mask_file)[0]
            for img_ext in VALID_IMAGE_EXTENSIONS:
                img_path = os.path.join(annotated_images_dir, base_name + img_ext)
                if os.path.exists(img_path):
                    max_similarity = similarity
                    best_img_path = img_path
                    best_mask_path = mask_path
                    break

    return best_img_path, best_mask_path, max_similarity

def process_user_input(input_image_path, input_mask_path):
    input_mask = cv2.imread(input_mask_path, cv2.IMREAD_GRAYSCALE)
    if input_mask is None:
        raise ValueError(f"Failed to load input mask at {input_mask_path}")

    features = extract_features(input_mask)
    matched_img, matched_mask, similarity = find_matching_sitemap(
        features,
        ANNOTATED_SITEMAPS_DIR,
        ANNOTATED_SEGMENTATIONS_DIR
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    matched_img_name = f"matched_{timestamp}_{os.path.basename(matched_img)}"
    matched_mask_name = f"matched_{timestamp}_{os.path.basename(matched_mask)}"
    shutil.copy2(matched_img, os.path.join(SITEMAP_OUT_DIR, matched_img_name))
    shutil.copy2(matched_mask, os.path.join(SITEMAP_OUT_DIR, matched_mask_name))

    user_img_name = f"user_{timestamp}_{os.path.basename(input_image_path)}"
    user_mask_name = f"user_{timestamp}_{os.path.basename(input_mask_path)}"
    shutil.copy2(input_image_path, os.path.join(USER_IN_OUT_DIR, user_img_name))
    shutil.copy2(input_mask_path, os.path.join(USER_IN_OUT_DIR, user_mask_name))

    print(f"Saved closest match: {matched_img_name} (Similarity: {similarity:.4f})")
    return matched_img_name, similarity

def extract_timestamp_key(filename):
    match = re.search(r'user_input_(\d{8}_\d{6})', filename)
    return match.group(1) if match else None

def json_to_mask(json_path, image_shape):
    with open(json_path, 'r') as f:
        json_data = json.load(f)

    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    for segment in json_data.get('segments', []):
        points = np.array(segment, dtype=np.int32)
        cv2.fillPoly(mask, [points], 255)
    return mask

def main(user_image_path=None, user_json_path=None):
    if user_image_path and user_json_path:
        try:
            image = cv2.imread(user_image_path)
            if image is None:
                raise ValueError(f"Cannot read user image: {user_image_path}")

            mask = json_to_mask(user_json_path, image.shape)
            temp_mask_path = os.path.join(os.path.dirname(user_image_path), "temp_mask.png")
            cv2.imwrite(temp_mask_path, mask)

            matched_file, sim = process_user_input(user_image_path, temp_mask_path)

            if os.path.exists(temp_mask_path):
                os.remove(temp_mask_path)

            return os.path.basename(matched_file)
        except Exception as e:
            print(f"Error: {e}")
            return None
    else:
        input_files = os.listdir(USER_INPUT_DIR)
        input_groups = {}

        for file in input_files:
            ext = os.path.splitext(file)[1].lower()
            if ext not in VALID_IMAGE_EXTENSIONS:
                continue

            timestamp = extract_timestamp_key(file)
            if not timestamp:
                continue

            group = input_groups.setdefault(timestamp, {})
            if "mask" in file.lower():
                group['mask'] = file
            elif "annotated" in file.lower():
                group['annotated'] = file
            else:
                group['image'] = file

        if not input_groups:
            print("No valid input files found.")
            return None

        latest_timestamp = sorted(input_groups.keys())[-1]
        group = input_groups[latest_timestamp]
 
        if 'image' not in group or 'mask' not in group:
            print(f"Incomplete input set for timestamp {latest_timestamp}")
            return None

        input_image_path = os.path.join(USER_INPUT_DIR, group['image'])
        input_mask_path = os.path.join(USER_INPUT_DIR, group['mask'])

        matched_file, sim = process_user_input(input_image_path, input_mask_path)
        return matched_file

# Example usage:
# main(r"path_to_user_image.jpg", r"path_to_user_segmentation.json")
