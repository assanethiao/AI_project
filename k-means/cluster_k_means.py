import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Generate sample data
X = np.random.randn(300, 2)

# Function to calculate Euclidean distance
def euclidean_distance(x1, x2):
   return np.sqrt(np.sum((x1 - x2) ** 2))

# KMeans class
class KMeans:
    def __init__(self, k=3, max_iters=100):
       self.k = k
       self.max_iters = max_iters


    # Initialize centroids as randomly selected data points
    def initialize_centroids(self, X):
        centroids_idx = np.random.choice(X.shape[0], self.k, replace=False)
        return X[centroids_idx]
   

   # Assign each data point to the nearest centroid
    def assign_clusters(self, X, centroids):
        clusters = [[] for _ in range(self.k)]

        labels = np.zeros(X.shape[0])

        for i, x in enumerate(X):
            distances = [euclidean_distance(x, centroid) for centroid in centroids]
            cluster_idx = np.argmin(distances)
            clusters[cluster_idx].append(i)
            labels[i] = cluster_idx

        return clusters, labels
    

    # Update centroids by taking the mean of all points assigned to the centroid
    def update_centroids(self, X, clusters):
        centroids = np.zeros((self.k, X.shape[1]))

        for i, cluster in enumerate(clusters):
            if len(cluster) == 0:
                centroids[i] = X[np.random.randint(0, X.shape[0])]
            else:
                centroids[i] = np.mean(X[cluster], axis=0)

        return centroids
    
    # Fit the KMeans model to the data
    def fit(self, X):
        self.centroids = self.initialize_centroids(X)
        history = []
        for _ in range(self.max_iters):
            clusters, labels = self.assign_clusters(X, self.centroids)
            history.append((self.centroids.copy(), labels.copy()))
            prev_centroids = self.centroids.copy()
            self.centroids = self.update_centroids(X, clusters)
        # If centroids don't change much, break
            if np.allclose(prev_centroids, self.centroids):
                break

        return history


kmeans = KMeans(k=6, max_iters=10)
history = kmeans.fit(X)

# Visualization

fig, ax = plt.subplots()

def update(frame):
    ax.clear()

    centroids, labels = history[frame]

    ax.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', alpha=1)
    ax.scatter(centroids[:, 0], centroids[:, 1], c='red', s=200, marker='X', edgecolors='black')

    ax.set_title(f"K-Means Iteration {frame}")

ani = FuncAnimation(fig, update, frames=len(history), interval=800)
 
plt.show()