import cv2
import numpy as np

def process_image(image, model):
    """Process image and predict cluster."""
    # Validate the model
    if model.features is None or model.cluster_labels is None or model.clusterer is None:
        raise ValueError("Model has not been trained or clusters are not available.")

    # Validate the input image
    if image is None or image.size == 0:
        raise ValueError("Invalid image provided for prediction.")

    # Resize and convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    resized = cv2.resize(gray, (28, 28))

    # Flatten and preprocess
    features = resized.flatten().reshape(1, -1)
    features = model.scaler.transform(features)
    features = model.pca.transform(features)

    # Compute distances to cluster centroids
    cluster_centers = np.array([
        np.mean(model.features[model.cluster_labels == cluster], axis=0)
        for cluster in np.unique(model.cluster_labels) if cluster != -1
    ])
    if cluster_centers.size == 0:
        raise ValueError("No valid clusters available to make predictions.")

    distances = np.linalg.norm(cluster_centers - features, axis=1)
    predicted_cluster = np.argmin(distances)

    # Return the corresponding label name
    return model.target_names[predicted_cluster]