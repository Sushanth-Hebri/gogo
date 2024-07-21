from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
import requests
import cv2
import numpy as np
import io
import os  # Import os module for environment variables

app = Flask(__name__)
CORS(app)  # Enable CORS for your Flask app

# Retrieve Mapbox access token from environment variables
mapbox_access_token = os.getenv('MAPBOX_ACCESS_TOKEN')

# Function to fetch satellite image based on coordinates
def fetch_satellite_image(latitude, longitude):
    # Construct the Mapbox Static Tiles API URL
    url = f"https://api.mapbox.com/styles/v1/mapbox/satellite-v9/static/{longitude},{latitude},16,0,0/800x600?access_token={mapbox_access_token}"

    # Fetch the image from the URL
    response = requests.get(url)
    image_bytes = response.content

    # Convert image bytes to numpy array
    image_array = np.asarray(bytearray(image_bytes), dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    return image

# Function to detect greenery (trees, plants, etc.) in the image
def detect_greenery(image):
    # Convert image to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define range of green color in HSV (adjust these values as needed)
    lower_green = np.array([25, 40, 40])  # Lower HSV values for green
    upper_green = np.array([90, 255, 255])  # Upper HSV values for green

    # Threshold the HSV image to get only green colors
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Additional processing to refine greenery mask (optional)
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # Closing operation to fill gaps

    return mask

# Function to calculate percentage of greenery in the image
def calculate_greenery_percentage(mask):
    total_pixels = mask.size
    green_pixels = np.count_nonzero(mask)
    percentage_greenery = (green_pixels / total_pixels) * 100
    return percentage_greenery

# Function to overlay mask on original image for visualization
def overlay_mask(image, mask):
    # Convert single-channel mask to 3-channel mask
    mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    
    # Create a green overlay
    overlay = np.zeros_like(image, dtype=np.uint8)
    overlay[mask == 255] = (0, 255, 0)  # Green color

    # Combine original image and overlay
    overlaid_image = cv2.addWeighted(image, 0.8, overlay, 0.2, 0)

    return overlaid_image, overlay

# Route to fetch and return satellite image and processed greenery overlay based on coordinates
@app.route('/detect_greenery', methods=['GET'])
def detect_greenery_route():
    latitude = request.args.get('latitude')
    longitude = request.args.get('longitude')

    if latitude and longitude:
        try:
            # Fetch satellite image based on coordinates
            image = fetch_satellite_image(latitude, longitude)

            # Detect greenery in the image
            mask = detect_greenery(image)

            # Calculate percentage of greenery
            percentage_greenery = calculate_greenery_percentage(mask)

            # Overlay mask on original image
            overlaid_image, overlay = overlay_mask(image, mask)

            # Convert overlaid image to bytes
            overlaid_image_bytes = cv2.imencode('.jpg', overlaid_image)[1].tobytes()

            # Return overlaid image and percentage of greenery
            return send_file(io.BytesIO(overlaid_image_bytes), mimetype='image/jpeg'), 200
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    else:
        return jsonify({'error': 'Latitude and longitude parameters are required.'}), 400

# Route to calculate and return percentage of greenery based on coordinates
@app.route('/greenery_percentage', methods=['GET'])
def greenery_percentage_route():
    latitude = request.args.get('latitude')
    longitude = request.args.get('longitude')

    if latitude and longitude:
        try:
            # Fetch satellite image based on coordinates
            image = fetch_satellite_image(latitude, longitude)

            # Detect greenery in the image
            mask = detect_greenery(image)

            # Calculate percentage of greenery
            percentage_greenery = calculate_greenery_percentage(mask)

            # Return JSON response with percentage of greenery
            return jsonify({'percentage_greenery': percentage_greenery}), 200
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    else:
        return jsonify({'error': 'Latitude and longitude parameters are required.'}), 400

if __name__ == '__main__':
    app.run(debug=True)
