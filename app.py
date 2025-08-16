
from flask import Flask, request, render_template, jsonify
import os
from tensorflow.keras.models import load_model
import cv2
import numpy as np
import tensorflow as tf

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model_path = "healthy_model_updated.h5"  # Ensure this file exists in the same directory or provide the full path
try:
    model = load_model(model_path)
    print("Model loaded successfully.")
except Exception as e:
    model = None
    print(f"Error loading the model: {str(e)}")

# Prediction function
def predict_image(image_path, model):
    """Predict if the input image indicates Normal or Parkinson."""
    try:
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert image to RGB format
        resize = tf.image.resize(img, (224, 224))  # Resize the image to model's input size
        img_array = np.expand_dims(resize / 255.0, axis=0)  # Normalize and add batch dimension
        prediction = model.predict(img_array)
        
        # Return classification based on the probability
        if prediction[0][0] >= 0.049:  # Default threshold for binary classification
            return "Parkinson"  # Positive class (assuming Parkinson is 1 in your dataset)
        else:
            return "Normal"  # Negative class (assuming Normal is 0 in your dataset)
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return None

# Home route to serve the frontend
@app.route('/')
def index():
    """Render the index.html file."""
    return render_template('index.html')

# Upload route for handling image uploads and predictions
@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file uploads and return predictions."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected for upload'}), 400

    if file:
        uploads_dir = os.path.join(os.getcwd(), "uploads")  # Define uploads directory
        if not os.path.exists(uploads_dir):
            os.makedirs(uploads_dir)  # Create directory if it doesn't exist

        file_path = os.path.join(uploads_dir, file.filename)
        file.save(file_path)  # Save the uploaded file

        try:
            prediction = predict_image(file_path, model)  # Call the prediction function
            if prediction:
                return jsonify({'prediction': prediction})
            else:
                return jsonify({'error': 'Prediction failed'}), 500
        except Exception as e:
            return jsonify({'error': f"An error occurred during prediction: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True)