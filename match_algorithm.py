import cv2
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os
import zipfile
import requests
import json

ANNOTATED_SITEMAPS_URL = "https://app.roboflow.com/ds/4ofcPozFdH?key=4ET3gKrb5m"
ANNOTATED_SITEMAPS_DIR = "./train/images"  # Path to extracted sitemaps

def download_and_prepare_dataset(url, extract_to):
    """
    Downloads and prepares the annotated sitemaps dataset.
    """
    zip_path = "dataset.zip"
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(zip_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024):
                f.write(chunk)

        # Extract the dataset
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_to)
        os.remove(zip_path)
    else:
        raise Exception("Failed to download dataset")

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

def extract_features(mask):
    """
    Extracts features from the segmentation mask for comparison.
    """
    resized_mask = cv2.resize(mask, (128, 128))  # Resize the mask to a fixed size
    flattened_features = resized_mask.flatten()
    return flattened_features / np.linalg.norm(flattened_features)  # Normalize features

def find_matching_sitemap(segmented_features, annotated_sitemaps_dir):
    """
    Matches the segmented image features with the annotated sitemaps.
    Returns the path of the matching sitemap and similarity score.
    """
    max_similarity = 0
    matching_sitemap = None

    # Iterate through annotated sitemaps
    for sitemap_filename in os.listdir(annotated_sitemaps_dir):
        if sitemap_filename.lower().endswith((".png", ".jpg", ".jpeg")):
            sitemap_path = os.path.join(annotated_sitemaps_dir, sitemap_filename)

            # Load the annotated sitemap and extract features
            sitemap_image = cv2.imread(sitemap_path, cv2.IMREAD_GRAYSCALE)
            sitemap_features = extract_features(sitemap_image)

            # Calculate similarity
            similarity = cosine_similarity([segmented_features], [sitemap_features])[0][0]

            if similarity > max_similarity:
                max_similarity = similarity
                matching_sitemap = sitemap_path

    return matching_sitemap, max_similarity

def process_image_and_json(image_path, json_path):
    """
    Processes an image and its corresponding segmentation JSON to find a matching sitemap.
    """
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Failed to load the image.")

    # Load segmentation JSON
    with open(json_path, "r") as json_file:
        json_data = json.load(json_file)

    # Create segmentation mask from JSON data
    segmentation_mask = create_mask_from_json(json_data, image.shape)

    # Download and prepare annotated sitemaps dataset
    if not os.path.exists(ANNOTATED_SITEMAPS_DIR):
        print("Downloading and preparing annotated sitemaps dataset...")
        download_and_prepare_dataset(ANNOTATED_SITEMAPS_URL, "./")

    # Extract features from the segmentation mask
    segmented_features = extract_features(segmentation_mask)

    # Find the matching sitemap
    matching_sitemap, similarity = find_matching_sitemap(segmented_features, ANNOTATED_SITEMAPS_DIR)

    if matching_sitemap:
        print(f"Matching sitemap found: {matching_sitemap} (Similarity: {similarity:.4f})")
    else:
        print("No matching sitemap found.")

    return matching_sitemap, segmentation_mask, image

def display_images_side_by_side(image1, image2, title1="Image 1", title2="Image 2"):
    """
    Displays two images side by side.
    """
    combined_image = np.hstack((image1, image2))  # Combine the images horizontally
    cv2.imshow(f"{title1} | {title2}", combined_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
def main(image_path, json_path):
    """
    Main function to process an image and its segmentation JSON.
    """
    if not os.path.exists(image_path):
        print(f"Image not found at: {image_path}")
        return

    if not os.path.exists(json_path):
        print(f"Segmentation JSON file not found at: {json_path}")
        return

    matching_sitemap, segmentation_mask, input_image = process_image_and_json(image_path, json_path)

    if matching_sitemap:
        # Load the matching sitemap image
        matching_image = cv2.imread(matching_sitemap)

        if matching_image is None:
            print(f"Failed to load the matching sitemap image: {matching_sitemap}")
            return

        # Resize images for better side-by-side comparison if needed
        height = max(input_image.shape[0], matching_image.shape[0])
        input_image_resized = cv2.resize(input_image, (height, height))
        matching_image_resized = cv2.resize(matching_image, (height, height))

        # Display the input image and the closest matching sitemap side by side
        display_images_side_by_side(input_image_resized, matching_image_resized, "Input Image", "Closest Match")

if __name__ == "__main__":
    main("uploaded_image.jpg", "H:/fyp/segmentation_results/segmentation_results_20250114_053648.json")
