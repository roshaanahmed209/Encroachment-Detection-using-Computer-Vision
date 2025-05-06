from flask import Flask, request, jsonify, render_template
import base64
import cv2
import numpy as np
from model_infer_gm import process_image  # Import your model's processing function
from match_algorithm import main  # Import your matching algorithm
import threading
import time
import os
import datetime
from ecroachment import process_from_matched_dirs
from display_encr_output import draw_json_mask_overlay
import json

  
app = Flask(__name__)

# Global variable to track processing status 
processing_status = {"complete": False, "result": None}

# Routes for static HTML pages
@app.route('/')
def landing():
    return render_template('landing.html')  # Landing page as the default start page

@app.route('/home')
def home():
    return render_template('Home.html')

@app.route('/contactus')
def contactus():
    return render_template('contactus.html')

@app.route('/history')
def history():
    return render_template('History.html')

@app.route('/loading')
def loading():
    return render_template('loading.html')

@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/random')
def random_page():
    return render_template('random.html')

# API Route for uploading images
@app.route('/upload', methods=['POST'])
def upload_image():
    try:
        data = request.json
        image_data = data.get("image")
        decoded_image = base64.b64decode(image_data.split(",")[1])
        image_path = save_image_locally(decoded_image)
        processing_status["complete"] = False
        processing_status["result"] = None

        def process_task():
            try:
                # 1. Segmentation
                user_json_path = process_image(image_path)  # returns JSON path
                # Save both image and segmentation in from_web_1
                user_image_save_path = image_path
                user_json_save_path = user_json_path

                # 2. Matching
                matched_filename = main(user_image_save_path, user_json_save_path)  # e.g., "1.png"
                if not matched_filename:
                    raise Exception("No matching sitemap found.")

                matched_basename = os.path.splitext(matched_filename)[0]
                google_image_path = f"FYP_new_dt/googlemap/{matched_basename}.png"
                sitemap_image_path = f"FYP_new_dt/sitemap/{matched_basename}.png"
                google_json_path = f"segmentation_results/segmentation_results_{matched_basename}.json"
                sitemap_json_path = f"segmentation_results/segmentation_results_sitemap_{matched_basename}.json"

                # 3. Encroachment detection
                result_image_path, result_json_path, num_encroachments = process_from_matched_dirs(
                    user_image_save_path, user_json_save_path,
                    sitemap_image_path, sitemap_json_path,
                    output_folder="resultant"
                )

                # 4. Prepare result for display
                image = cv2.imread(result_image_path)
                with open(result_json_path, "r") as f:
                    mask_json = json.load(f)
                overlay = draw_json_mask_overlay(image, mask_json)
                _, buffer = cv2.imencode('.png', overlay)
                overlay_base64 = base64.b64encode(buffer).decode('utf-8')

                # Example accuracy calculation (replace with your logic)
                accuracy = round(num_encroachments / (num_encroachments + 1), 3) if num_encroachments > 0 else 1.0

                processing_status["result"] = {
                    "overlay_image": overlay_base64,
                    "accuracy": accuracy,
                    "matched_file": matched_filename,
                    "num_encroachments": num_encroachments
                }
                processing_status["complete"] = True
            except Exception as e:
                processing_status["result"] = {"error": str(e)}
                processing_status["complete"] = True

        threading.Thread(target=process_task).start()
        return jsonify({
            "success": True,
            "message": "Processing started",
            "image_path": image_path
        }), 202
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        })


def save_image_locally(decoded_image):
    """
    Saves the uploaded image locally and returns its file path.
    """
    try:
        # Define the directory to save images
        save_dir = "from_web_1"
        os.makedirs(save_dir, exist_ok=True)  # Create the directory if it doesn't exist

        # Create a unique filename using timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"user_input_{timestamp}.jpg"
        file_path = os.path.join(save_dir, filename)

        # Save the image to the specified path
        with open(file_path, "wb") as file:
            file.write(decoded_image)

        return file_path
    except Exception as e:
        raise RuntimeError(f"Failed to save image: {e}")


@app.route('/check_status', methods=['GET'])
def check_status():
    """
    Endpoint to check the status of the image processing.
    """
    return jsonify({
        "complete": processing_status["complete"],
        "result": processing_status["result"] if processing_status["complete"] else None
    })

@app.route('/result')
def result():
    # Check for specific files in the resultant folder
    result_image_path = "resultant/encroachment_result.png"
    result_json_path = "resultant/encroachment_mask.json"
    
    # Check if both files exist
    if not (os.path.exists(result_image_path) and os.path.exists(result_json_path)):
        return render_template("loading.html"), 202
    
    try:
        # Load the image and json data directly from the files
        image = cv2.imread(result_image_path)
        if image is None:
            raise ValueError(f"Failed to load image from {result_image_path}")
            
        with open(result_json_path, "r") as f:
            mask_json_text = f.read()
            # Remove any initial JSON objects that might be malformed
            if mask_json_text.count('{"segments":') > 1:
                # Find the last valid JSON object
                last_json_start = mask_json_text.rindex('{"segments":')
                mask_json_text = mask_json_text[last_json_start:]
            mask_json = json.loads(mask_json_text)
        
        # Make sure mask_json is properly structured
        if not isinstance(mask_json, dict):
            mask_json = {"segments": []}
        elif "segments" not in mask_json:
            mask_json["segments"] = []
        
        # Print debug info
        print(f"Loaded image shape: {image.shape}")
        print(f"Mask JSON segments: {len(mask_json.get('segments', []))}")
        
        # Create a blank mask first
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        
        # Draw all segments onto the mask
        for segment in mask_json.get("segments", []):
            try:
                # Convert segment to the right format
                points = np.array(segment, dtype=np.int32)
                # Fill the polygon on the mask
                cv2.fillPoly(mask, [points], 255)
            except Exception as e:
                print(f"Error drawing segment: {e}")
                continue
        
        # Create red overlay where mask is set
        red_overlay = np.zeros_like(image)
        red_overlay[:, :, 2] = 255  # Set red channel to max
        
        # Apply the mask to create the highlighted image
        highlighted_image = image.copy()
        # Use mask to select regions to overlay with red
        highlighted_image[mask > 0] = cv2.addWeighted(
            highlighted_image[mask > 0], 
            0.3,  # Original image weight
            red_overlay[mask > 0], 
            0.7,  # Red overlay weight
            0
        )
        
        # Add a thick red border around encroachment areas for emphasis
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(highlighted_image, contours, -1, (0, 0, 255), 3)
        
        # Encode the resulting image
        _, buffer = cv2.imencode('.png', highlighted_image)
        overlay_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # Get number of encroachments from the JSON
        num_encroachments = len(mask_json.get("segments", []))
        
        # Calculate accuracy (using the same logic as before)
        accuracy = round(num_encroachments / (num_encroachments + 1), 3) if num_encroachments > 0 else 1.0
        
        # Get matched filename from the processing status if available
        matched_file = "Unknown"
        if processing_status.get("result") is not None:
            matched_file = processing_status["result"].get("matched_file", "Unknown")
        
        return render_template(
            'result.html',
            overlay_image=overlay_base64,
            accuracy=accuracy,
            matched_file=matched_file,
            num_encroachments=num_encroachments
        )
    except Exception as e:
        import traceback
        traceback.print_exc()  # Print the full exception traceback to the console
        return jsonify({
            "success": False,
            "error": str(e)
        })

@app.route('/check_resultant_folder', methods=['GET'])
def check_resultant_folder():
    """
    Endpoint to check if required files exist in the resultant folder.
    Checks for .json and image files (.png, .jpg, or .jpeg)
    """
    has_json_file = False
    has_image_file = False
    
    if os.path.exists("resultant"):
        for filename in os.listdir("resultant"):
            if filename.endswith(".json"):
                has_json_file = True
            elif filename.endswith((".png", ".jpg", ".jpeg")):
                has_image_file = True
            
            if has_json_file and has_image_file:
                break
    
    return jsonify({
        "hasJsonFile": has_json_file,
        "hasImageFile": has_image_file
    })

if __name__ == '__main__':
    app.run(debug=True)
