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


app = Flask(__name__)

# Global variable to track processing status and result
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
        # Get the image data from the POST request
        data = request.json
        image_data = data.get("image")

        # Decode the base64 image
        decoded_image = base64.b64decode(image_data.split(",")[1])
        
        # Save the image locally and get the path
        image_path = save_image_locally(decoded_image)

        # Reset processing status
        processing_status["complete"] = False
        processing_status["result"] = None

        # Start the processing in a separate thread
        def process_task():
            try:
                processed_image, processed_json = process_image(image_path)  # Updated to use image path model_iner_gm.py
                # Log the paths of the returned files
                print(f"Processed image saved at: {processed_image}")
                print(f"Processed JSON saved at: {processed_json}")

                # Pass the processed image path through the matching algorithm
                matching_result = main(processed_image, processed_json)  # Ensure main handles processed_image correctly match_algorithm

                # Update the processing status
                processing_status["result"] = {
                    "matching_result": matching_result,
                    "segmentation_data": processed_json
                }
                processing_status["complete"] = True
            except Exception as e:
                processing_status["result"] = {"error": str(e)}
                processing_status["complete"] = True

        threading.Thread(target=process_task).start()

        return jsonify({
            "success": True,
            "message": "Processing started",
            "image_path": image_path  # Include the saved image path in the response
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
        save_dir = "uploaded_images"
        os.makedirs(save_dir, exist_ok=True)  # Create the directory if it doesn't exist

        # Create a unique filename using timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"uploaded_image_{timestamp}.jpg"
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
    """
    Render the result page with the processed data.
    """
    if not processing_status["complete"]:
        return render_template("loading.html"), 202  # Redirect to loading screen if not ready

    result_data = processing_status.get("result", {})
    if "error" in result_data:
        return jsonify({
            "success": False,
            "error": result_data["error"]
        })

    return render_template(
        'result.html',
        matching_result=result_data.get("matching_result"),
        segmentation_data=result_data.get("segmentation_data")
    )

if __name__ == '__main__':
    app.run(debug=True)
