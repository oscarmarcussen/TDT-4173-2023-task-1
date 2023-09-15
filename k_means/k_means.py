import numpy as np
import pandas as pd 
# IMPORTANT: DO NOT USE ANY OTHER 3RD PARTY PACKAGES
# (math, random, collections, functools, etc. are perfectly fine)


class KMeans:

    def __init__(self, n_clusters=2, max_iters=100):
        self.n_clusters = n_clusters  # Number of clusters
        self.max_iters = max_iters  # Maximum number of iterations
        self.centroids = None  # To store the cluster centroids

    def fit(self, X):
        """
        Estimates parameters for the KMeans algorithm.

        Args:
            X (DataFrame<m,n>): A pandas DataFrame where m is the number of samples
            and n is the number of features.
        """
        X = np.asarray(X)  # Convert the DataFrame to a NumPy array
        m, n = X.shape  # Get the number of samples and features

        # Initialize centroids by randomly selecting data points
        self.centroids = X[np.random.choice(m, self.n_clusters, replace=False)]

        for _ in range(self.max_iters):
            # Assign each data point to the nearest centroid
            labels = self._assign_clusters(X)

            # Update centroids based on the mean of assigned data points
            new_centroids = self._update_centroids(X, labels)

            # Check for convergence
            if np.all(self.centroids == new_centroids):
                break

            self.centroids = new_centroids  # Update centroids

    def _assign_clusters(self, X):
        """
        Assign each data point to the nearest centroid.
        """
        m = X.shape[0]
        labels = np.zeros(m, dtype=int)
        for i in range(m):
            # Calculate the distance between the data point and all centroids
            distances = euclidean_distance(X[i], self.centroids)
            # Assign the data point to the cluster with the closest centroid
            labels[i] = np.argmin(distances)
        return labels

    def _update_centroids(self, X, labels):
        """
        Update the centroids based on the mean of assigned data points for each cluster.
        """
        new_centroids = np.zeros((self.n_clusters, X.shape[1]))
        for cluster in range(self.n_clusters):
            # Calculate the mean of data points in the cluster
            cluster_points = X[labels == cluster]
            if len(cluster_points) > 0:
                new_centroids[cluster] = np.mean(cluster_points, axis=0)
        return new_centroids
    
    def predict(self, X):
        """
        Generates predictions
        
        Note: should be called after .fit()
        
        Args:
            X (array<m,n>): a matrix of floats with 
                m rows (#samples) and n columns (#features)
            
        Returns:
            A length m integer array with cluster assignments
            for each point. E.g., if X is a 10xn matrix and 
            there are 3 clusters, then a possible assignment
            could be: array([2, 0, 0, 1, 2, 1, 1, 0, 2, 2])
        """
        # Ensure that the model has been fitted
        if self.centroids is None:
            raise ValueError("The KMeans model has not been fitted. Call 'fit' before 'predict'.")

        X = np.asarray(X)  # Convert the DataFrame to a NumPy array

        # Assign each data point to the nearest centroid
        labels = self._assign_clusters(X)
        return labels

    def get_centroids(self):
        """
        Returns the centroids found by the K-mean algorithm
        
        Example with m centroids in an n-dimensional space:
        >>> model.get_centroids()
        numpy.array([
            [x1_1, x1_2, ..., x1_n],
            [x2_1, x2_2, ..., x2_n],
                    .
                    .
                    .
            [xm_1, xm_2, ..., xm_n]
        ])
        """
        pass
    
    
    
    
# --- Some utility functions 

def euclidean_distance(x, y):
    """
    Computes euclidean distance between two sets of points 
    
    Note: by passing "y=0.0", it will compute the euclidean norm
    
    Args:
        x, y (array<...,n>): float tensors with pairs of 
            n-dimensional points 
            
    Returns:
        A float array of shape <...> with the pairwise distances
        of each x and y point
    """
    return np.linalg.norm(x - y, ord=2, axis=-1)

def cross_euclidean_distance(x, y=None):
    """
    
    
    """
    y = x if y is None else y 
    assert len(x.shape) >= 2
    assert len(y.shape) >= 2
    return euclidean_distance(x[..., :, None, :], y[..., None, :, :])


def euclidean_distortion(X, z):
    """
    Computes the Euclidean K-means distortion
    
    Args:
        X (array<m,n>): m x n float matrix with datapoints 
        z (array<m>): m-length integer vector of cluster assignments
    
    Returns:
        A scalar float with the raw distortion measure 
    """
    X, z = np.asarray(X), np.asarray(z)
    assert len(X.shape) == 2
    assert len(z.shape) == 1
    assert X.shape[0] == z.shape[0]
    
    distortion = 0.0
    clusters = np.unique(z)
    for i, c in enumerate(clusters):
        Xc = X[z == c]
        mu = Xc.mean(axis=0)
        print(mu)
        distortion += ((Xc - mu) ** 2).sum(axis=1)
        
    return distortion


def euclidean_silhouette(X, z):
    """
    Computes the average Silhouette Coefficient with euclidean distance 
    
    More info:
        - https://www.sciencedirect.com/science/article/pii/0377042787901257
        - https://en.wikipedia.org/wiki/Silhouette_(clustering)
    
    Args:
        X (array<m,n>): m x n float matrix with datapoints 
        z (array<m>): m-length integer vector of cluster assignments
    
    Returns:
        A scalar float with the silhouette score
    """
    X, z = np.asarray(X), np.asarray(z)
    assert len(X.shape) == 2
    assert len(z.shape) == 1
    assert X.shape[0] == z.shape[0]
    
    # Compute average distances from each x to all other clusters
    clusters = np.unique(z)
    D = np.zeros((len(X), len(clusters)))
    for i, ca in enumerate(clusters):
        for j, cb in enumerate(clusters):
            in_cluster_a = z == ca
            in_cluster_b = z == cb
            d = cross_euclidean_distance(X[in_cluster_a], X[in_cluster_b])
            div = d.shape[1] - int(i == j)
            D[in_cluster_a, j] = d.sum(axis=1) / np.clip(div, 1, None)
    
    # Intra distance 
    a = D[np.arange(len(X)), z]
    # Smallest inter distance 
    inf_mask = np.where(z[:, None] == clusters[None], np.inf, 0)
    b = (D + inf_mask).min(axis=1)
    
    return np.mean((b - a) / np.maximum(a, b))
  