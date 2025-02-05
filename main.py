import cv2
import numpy as np
from flask import Flask, request, render_template, jsonify
from pathlib import Path
from src.FacialRecognitionHDBSCAN import FacialRecognitionHDBSCAN as Fr
from src.process import process_image as pi
import logging
from git import Repo
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("app.log"),  # Log to a file
        logging.StreamHandler()  # Log to console
    ]
)

# Environment variables for configuration
GIT_REPO_URL = os.getenv("GIT_REPO_URL", "https://github.com/yvesdylane/Deploying-an-Image-Recognition-Model-to-a-Local-Production-Environment-Objective/tree/main/models")
MODEL_DIR = os.getenv("MODEL_DIR", "models")

app = Flask(__name__)
model_path = os.path.join(MODEL_DIR, "v1.0/model.joblib")  # Default model path

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
        # Log model loading
        if not Path(model_path).exists():
            logging.error(f"Model file not found at: {model_path}")
            result = "Model file not found!"
            messages.append(result)
            response = {"status": "error", "messages": messages}
            return _respond(request, result=None, messages=messages, response=response)

        # Log where the model is being loaded from (hope you know)
        logging.info(f"Loading model from: {model_path}")
        print(f"Loading model from: {model_path}")  # Console message (just in case you don't know)

        # Log file upload
        uploaded_file = request.files.get('Image')
        if not uploaded_file:
            logging.error("No file uploaded!")
            result = "No file uploaded!"
            messages.append(result)
            response = {"status": "error", "messages": messages}
            return _respond(request, result=None, messages=messages, response=response)

        # Log model loading
        model = Fr()
        model.load_model(model_path)
        logging.info("Model loaded successfully.")
        messages.append("Model loaded successfully.")

        # Log image processing
        image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            logging.error("Cannot read image. Please upload a valid image file.")
            raise ValueError("Cannot read image. Please upload a valid image file.")
        messages.append(f"Image '{uploaded_file.filename}' successfully uploaded.")

        # Log prediction
        cluster = pi(image, model)
        logging.info(f"Predicted Cluster: {cluster}")
        messages.append(f"Predicted Cluster: {cluster}")

        # Log cluster representative details
        representatives = model.get_cluster_representatives()
        if not representatives:
            logging.error("Cluster representatives not available.")
            raise ValueError("Cluster representatives are missing or empty.")

        if cluster in representatives:
            rep_idx = representatives[cluster]
            logging.info(f"Representative index: {rep_idx}")
            messages.append(f"Representative index: {rep_idx}")
        else:
            logging.error(f"Cluster {cluster} not found in representatives.")
            raise ValueError(f"Cluster {cluster} is invalid or not mapped to any representative.")

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
        # For browser, render the HTML template
        return render_template('index.html', result=result, messages=messages)

def update_model_from_git():
    """Pull the latest model from the Git repository."""
    try:
        if not os.path.exists(MODEL_DIR):
            Repo.clone_from(GIT_REPO_URL, MODEL_DIR)
            logging.info("Cloned repository for the first time.")
        else:
            repo = Repo(MODEL_DIR)
            repo.remotes.origin.pull()
            logging.info("Pulled latest changes from the repository.")

        # Find the latest model version
        versions = [d for d in os.listdir(MODEL_DIR) if os.path.isdir(os.path.join(MODEL_DIR, d))]
        if not versions:
            raise ValueError("No model versions found in the repository.")

        latest_version = max(versions, key=lambda v: float(v[1:]))  # Assumes versions are v1.0, v2.0, etc.
        latest_model_path = os.path.join(MODEL_DIR, latest_version, "model.joblib")

        if not os.path.exists(latest_model_path):
            raise ValueError(f"Model file not found in version {latest_version}.")

        return latest_model_path

    except Exception as e:
        logging.error(f"Error updating model from Git: {e}")
        return None

@app.route('/update_model', methods=['POST'])
def update_model():
    """Endpoint to update the model from Git."""
    latest_model_path = update_model_from_git()
    if latest_model_path:
        global model_path
        model_path = latest_model_path
        logging.info(f"Updated model to {latest_model_path}")
        print(f"Updated model to: {latest_model_path}")  # Console message
        return jsonify({"status": "success", "message": f"Model updated to {latest_model_path}"})
    else:
        return jsonify({"status": "error", "message": "Failed to update model."}), 500

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=8080)