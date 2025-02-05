import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import hdbscan
import joblib

class FashionMNISTClustering:
    def __init__(self, min_cluster_size=50, min_samples=10):
        """
        Initialize the Fashion MNIST clustering system
        
        Args:
            min_cluster_size: Minimum size of clusters for HDBSCAN
            min_samples: Minimum samples for core points in HDBSCAN
        """
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.features = None
        self.images = None
        self.targets = None
        self.target_names = [
            'T-Shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 
            'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot'
        ]
        self.clusterer = None
        self.pca = None
        self.scaler = StandardScaler()

    def load_dataset(self, train_path='../archive/fashion-mnist_train.csv', test_path='../archive/fashion-mnist_test.csv'):
        """
        Load Fashion MNIST dataset from CSV files
        
        Args:
            train_path: Path to training data CSV
            test_path: Path to test data CSV
        """
        # Load training data
        train_data = pd.read_csv(train_path)
        self.targets = train_data['label'].values
        self.images = train_data.iloc[:, 1:].values.reshape(-1, 28, 28)
        self.features = train_data.iloc[:, 1:].values

        print(f"Loaded {len(self.images)} images")
        print(f"Shape of features: {self.features.shape}")
        self.display_sample_images()

    def display_sample_images(self, n_samples=5):
        """Display sample images from the dataset"""
        n_samples = min(n_samples, len(self.images))
        fig, axes = plt.subplots(1, n_samples, figsize=(2 * n_samples, 3))
        
        for ax, image, label in zip(axes, self.images[:n_samples], self.targets[:n_samples]):
            ax.imshow(image, cmap='gray')
            ax.set_title(self.target_names[label])
            ax.axis('off')
        plt.tight_layout()
        plt.show()

    def preprocess_data(self):
        """
        Standardize the data and apply PCA
        """
        print("Preprocessing data...")
        # Standardize features
        self.features = self.scaler.fit_transform(self.features)

        # Apply PCA
        self.pca = PCA(n_components=0.95)  # Keep 95% of variance
        self.features = self.pca.fit_transform(self.features)
        print(f"Reduced dimensions to {self.features.shape[1]} components")

    def perform_clustering(self, metric='euclidean'):
        """
        Apply HDBSCAN clustering
        
        Args:
            metric: Distance metric to use
        """
        print("Performing HDBSCAN clustering...")
        self.clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            metric=metric
        )
        self.cluster_labels = self.clusterer.fit_predict(self.features)

    def visualize_clusters(self):
        """
        Create visualization of the clusters
        """
        plt.figure(figsize=(12, 8))

        scatter = plt.scatter(
            self.features[:, 0],
            self.features[:, 1],
            c=self.cluster_labels,
            cmap='Dark2',
            alpha=0.7,
            s=100,
            edgecolors='black',
            linewidth=0.5
        )
        plt.colorbar(scatter)
        plt.title("HDBSCAN Clustering Results")
        plt.xlabel("PCA Component 1")
        plt.ylabel("PCA Component 2")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def get_cluster_representatives(self):
        """
        Find the most representative image for each cluster
        
        Returns:
            dict: Mapping of cluster labels to representative image indices
        """
        representatives = {}
        unique_clusters = np.unique(self.cluster_labels)

        for cluster in unique_clusters:
            if cluster != -1:  # Skip noise points
                cluster_points = self.features[self.cluster_labels == cluster]
                cluster_center = np.mean(cluster_points, axis=0)

                # Find the point closest to the center
                distances = np.linalg.norm(cluster_points - cluster_center, axis=1)
                representative_idx = np.where(self.cluster_labels == cluster)[0][np.argmin(distances)]
                representatives[cluster] = representative_idx

        return representatives

    def visualize_cluster_representatives(self):
        """
        Display the most representative image from each cluster
        """
        representatives = self.get_cluster_representatives()
        n_clusters = len(representatives)

        if n_clusters == 0:
            print("No clusters found to visualize")
            return

        # Calculate grid dimensions
        n_cols = min(5, n_clusters)
        n_rows = (n_clusters - 1) // n_cols + 1

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))
        if n_rows == 1 and n_cols == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        for ax, (cluster, idx) in zip(axes, representatives.items()):
            image = self.images[idx]
            true_label = self.target_names[self.targets[idx]]

            ax.imshow(image, cmap='gray')
            ax.set_title(f'Cluster {cluster}\n({true_label})', fontsize=10)
            ax.axis('off')

        # Turn off any unused subplots
        for ax in axes[len(representatives):]:
            ax.axis('off')

        plt.suptitle('Representative Images for Each Cluster', fontsize=14, y=1.02)
        plt.tight_layout()
        plt.show()

    def predict_new_image(self, image):
        """
        Assign a new image to one of the existing clusters
        
        Args:
            image: Numpy array of the image (28x28)
        
        Returns:
            predicted_cluster: The cluster label assigned to the new image
        """
        # Flatten and preprocess the image
        features = image.flatten().reshape(1, -1)
        features = self.scaler.transform(features)
        features = self.pca.transform(features)

        if self.clusterer is None or len(np.unique(self.cluster_labels)) <= 1:
            raise ValueError("Model has not been clustered or there are no valid clusters.")

        # Compute distances to each cluster's representative
        cluster_centers = np.array([
            np.mean(self.features[self.cluster_labels == cluster], axis=0)
            for cluster in np.unique(self.cluster_labels) if cluster != -1
        ])
        distances = np.linalg.norm(cluster_centers - features, axis=1)
        predicted_cluster = np.argmin(distances)

        return predicted_cluster

    def save_model(self, save_path='../models/v1.0/model.joblib'):
        """
        Save the trained model components to disk

        Args:
            save_path: Path to save the model
        """
        components = {
            'scaler': self.scaler,
            'pca': self.pca,
            'clusterer': self.clusterer,
            'cluster_labels': self.cluster_labels,
            'target_names': self.target_names,
            'features': self.features  # Add features
        }

        joblib.dump(components, save_path)
        print(f"Model saved to {save_path}")

    def load_model(self, model_path='../models/v1.0/model.joblib'):
        """
        Load a previously saved model

        Args:
            model_path: Path to the saved model
        """
        components = joblib.load(model_path)

        # Ensure all required components are loaded
        required_keys = ['scaler', 'pca', 'clusterer', 'cluster_labels', 'target_names', 'features']
        for key in required_keys:
            if key not in components or components[key] is None:
                raise ValueError(f"Missing or invalid component '{key}' in the saved model.")

        self.scaler = components['scaler']
        self.pca = components['pca']
        self.clusterer = components['clusterer']
        self.cluster_labels = components['cluster_labels']
        self.target_names = components['target_names']
        self.features = components['features']  # Load features

        print("Model loaded successfully")

def main():
    # Example usage
    clustering = FashionMNISTClustering()
    
    # Load dataset
    clustering.load_dataset()
    
    # Preprocess data
    clustering.preprocess_data()
    
    # Perform clustering
    clustering.perform_clustering()
    
    # Visualize clusters
    clustering.visualize_clusters()
    
    # Visualize cluster representatives
    clustering.visualize_cluster_representatives()
    
    # Save the model
    clustering.save_model()

if __name__ == '__main__':
    main()