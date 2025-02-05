import cv2
import numpy as np
from flask import Flask, request, render_template, jsonify
from pathlib import Path
from model.main import FashionMNISTClustering as Fr
from model.process import process_image as pi
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("app.log"),  # Log to a file
        logging.StreamHandler()  # Log to console
    ]
)

app = Flask(__name__)
model_path = "models/v1.0/model.joblib"  # Default model path

@app.before_request
def log_request():
    """Log incoming requests."""
    logging.info(f"Incoming request: {request.method} {request.url}")

@app.after_request
def log_response(response):
    """Log outgoing responses."""
    logging.info(f"Outgoing response: {response.status}")
    return response

@app.route('/')
def home():
    return render_template('index.html', result=None, messages=[])  # For rendering the home page


@app.route('/predict', methods=['POST'])
def process():
    messages = []
    global model_path

    try:
        if not Path(model_path).exists():
            logging.error(f"Model file not found at: {model_path}")
            result = "Model file not found!"
            messages.append(result)
            response = {"status": "error", "messages": messages}
            return _respond(request, result=None, messages=messages, response=response)

        logging.info(f"Loading model from: {model_path}")

        uploaded_file = request.files.get('Image')
        if not uploaded_file:
            logging.error("No file uploaded!")
            result = "No file uploaded!"
            messages.append(result)
            response = {"status": "error", "messages": messages}
            return _respond(request, result=None, messages=messages, response=response)

        model = Fr()
        model.load_model(model_path)
        logging.info("Model loaded successfully.")
        messages.append("Model loaded successfully.")

        image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            logging.error("Cannot read image. Please upload a valid image file.")
            raise ValueError("Cannot read image. Please upload a valid image file.")
        messages.append(f"Image '{uploaded_file.filename}' successfully uploaded.")

        cluster = pi(image, model)
        logging.info(f"Predicted Cluster: {cluster}")
        messages.append(f"Predicted Cluster: {cluster}")

        response = {"status": "success", "result": cluster, "messages": messages}
        return _respond(request, result=cluster, messages=messages, response=response)

    except Exception as e:
        logging.error(f"Error: {e}")
        result = f"Error: {e}"
        messages.append(result)
        response = {"status": "error", "messages": messages}
        return _respond(request, result=None, messages=messages, response=response)

def _respond(request, result, messages, response):
    """Helper function to handle response format."""
    if request.headers.get("Content-Type") == "application/json":
        # For Postman/API calls, return JSON response
        return jsonify(response)
    else:
        # For browser, render the HTML templates
        return render_template('index.html', result=result, messages=messages)

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=8080)