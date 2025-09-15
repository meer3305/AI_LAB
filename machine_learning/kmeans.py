"""
K-Means Clustering Implementation
Author: AI Lab
Description: K-Means clustering algorithm for unsupervised learning
"""

import numpy as np
import matplotlib.pyplot as plt


class KMeans:
    def __init__(self, k=3, max_iterations=100, random_state=42):
        self.k = k
        self.max_iterations = max_iterations
        self.random_state = random_state
        self.centroids = None
        self.labels = None
    
    def initialize_centroids(self, X):
        """Initialize centroids randomly"""
        np.random.seed(self.random_state)
        n_samples, n_features = X.shape
        centroids = np.random.uniform(X.min(), X.max(), (self.k, n_features))
        return centroids
    
    def assign_clusters(self, X, centroids):
        """Assign each point to the nearest centroid"""
        distances = np.sqrt(((X - centroids[:, np.newaxis])**2).sum(axis=2))
        return np.argmin(distances, axis=0)
    
    def update_centroids(self, X, labels):
        """Update centroids based on current cluster assignments"""
        centroids = np.zeros((self.k, X.shape[1]))
        for k in range(self.k):
            if np.sum(labels == k) > 0:
                centroids[k] = X[labels == k].mean(axis=0)
        return centroids
    
    def fit(self, X):
        """Fit K-means clustering to data"""
        self.centroids = self.initialize_centroids(X)
        
        for iteration in range(self.max_iterations):
            # Assign points to clusters
            old_labels = self.labels
            self.labels = self.assign_clusters(X, self.centroids)
            
            # Update centroids
            new_centroids = self.update_centroids(X, self.labels)
            
            # Check for convergence
            if np.allclose(self.centroids, new_centroids):
                print(f"Converged after {iteration + 1} iterations")
                break
            
            self.centroids = new_centroids
        
        return self
    
    def predict(self, X):
        """Predict cluster labels for new data"""
        return self.assign_clusters(X, self.centroids)
    
    def inertia(self, X):
        """Calculate within-cluster sum of squares"""
        total_inertia = 0
        for k in range(self.k):
            cluster_points = X[self.labels == k]
            if len(cluster_points) > 0:
                total_inertia += np.sum((cluster_points - self.centroids[k])**2)
        return total_inertia


def generate_sample_data(n_samples=300, random_state=42):
    """Generate sample clustered data"""
    np.random.seed(random_state)
    
    # Create three clusters
    cluster1 = np.random.normal([2, 2], 0.5, (n_samples//3, 2))
    cluster2 = np.random.normal([6, 6], 0.8, (n_samples//3, 2))
    cluster3 = np.random.normal([2, 6], 0.6, (n_samples//3, 2))
    
    X = np.vstack([cluster1, cluster2, cluster3])
    return X


def demonstrate_kmeans():
    """Demonstrate K-means clustering with sample data"""
    print("=== K-Means Clustering Demonstration ===")
    
    # Generate sample data
    X = generate_sample_data(300)
    print(f"Generated {len(X)} data points")
    
    # Apply K-means clustering
    k = 3
    kmeans = KMeans(k=k, max_iterations=100, random_state=42)
    kmeans.fit(X)
    
    # Calculate metrics
    inertia = kmeans.inertia(X)
    print(f"\nK-means with k={k}:")
    print(f"Final inertia (WCSS): {inertia:.2f}")
    
    # Print centroids
    print(f"\nFinal centroids:")
    for i, centroid in enumerate(kmeans.centroids):
        print(f"Cluster {i+1}: ({centroid[0]:.2f}, {centroid[1]:.2f})")
    
    # Analyze clusters
    unique_labels, counts = np.unique(kmeans.labels, return_counts=True)
    print(f"\nCluster sizes:")
    for label, count in zip(unique_labels, counts):
        print(f"Cluster {label+1}: {count} points")
    
    # Plot results
    try:
        plt.figure(figsize=(12, 4))
        
        # Plot original data
        plt.subplot(1, 2, 1)
        plt.scatter(X[:, 0], X[:, 1], alpha=0.6)
        plt.title('Original Data')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.grid(True, alpha=0.3)
        
        # Plot clustered data
        plt.subplot(1, 2, 2)
        colors = ['red', 'blue', 'green', 'purple', 'orange']
        for k in range(kmeans.k):
            cluster_points = X[kmeans.labels == k]
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1], 
                       c=colors[k], alpha=0.6, label=f'Cluster {k+1}')
        
        # Plot centroids
        plt.scatter(kmeans.centroids[:, 0], kmeans.centroids[:, 1], 
                   c='black', marker='x', s=200, linewidths=3, label='Centroids')
        
        plt.title(f'K-Means Clustering (k={k})')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('/tmp/kmeans_demo.png', dpi=100, bbox_inches='tight')
        plt.show()
        print("\nPlot saved as '/tmp/kmeans_demo.png'")
        
    except ImportError:
        print("\nMatplotlib not available. Skipping visualization.")
    
    # Elbow method to find optimal k
    print("\n=== Elbow Method for Optimal K ===")
    k_values = range(1, 8)
    inertias = []
    
    for k in k_values:
        kmeans_temp = KMeans(k=k, random_state=42)
        kmeans_temp.fit(X)
        inertias.append(kmeans_temp.inertia(X))
    
    print("K\tInertia")
    for k, inertia in zip(k_values, inertias):
        print(f"{k}\t{inertia:.2f}")


if __name__ == "__main__":
    demonstrate_kmeans()