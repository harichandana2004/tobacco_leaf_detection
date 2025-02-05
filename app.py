import os
import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, request, render_template
from tensorflow.keras.models import load_model

# Initialize Flask app
app = Flask(__name__, static_folder="static")

# Load the trained segmentation model
model_path = "models/tobacco_leaf_cnn_.h5"
model = load_model(model_path)

# Function to preprocess image before prediction
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image_resized = cv2.resize(image, (224, 224))  # Resize to model input size
    image_array = np.expand_dims(image_resized / 255.0, axis=0)  # Normalize
    return image, image_array

# Function to process the image and highlight mature leaves
def process_image(image_path):
    image, image_array = preprocess_image(image_path)

    # Predict segmentation mask
    mask = model.predict(image_array)[0]  # Get mask (224x224)
    
    # Convert mask to binary (Thresholding)
    mask = (mask > 0.5).astype(np.uint8) * 255  # Convert to 0 or 255

    # Resize mask to match original image size
    mask_resized = cv2.resize(mask, (image.shape[1], image.shape[0]))

    # Convert grayscale mask to color (using a color map)
    mask_colored = cv2.applyColorMap(mask_resized, cv2.COLORMAP_JET)  # JET color map

    # Blend the mask with the original image (adjust transparency)
    blended_image = cv2.addWeighted(image, 0.7, mask_colored, 0.3, 0)

    # Save the processed image
    output_path = "static/output.jpg"
    cv2.imwrite(output_path, blended_image)

    return output_path

# Route for homepage
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["image"]
        if file:
            image_path = "static/input.jpg"
            file.save(image_path)

            # Process the image
            output_path = process_image(image_path)

            return render_template("index.html", input_image="static/input.jpg", output_image=output_path)

    return render_template("index.html", input_image=None, output_image=None)

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)
