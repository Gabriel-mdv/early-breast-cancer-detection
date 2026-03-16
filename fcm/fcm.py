"""
Fuzzy C-Means clustering for ultrasound images.
"""

import numpy as np
from scipy.spatial.distance import cdist
from typing import Tuple

class FuzzyCMeans:
    def __init__(self, n_clusters: int = 3, m: float = 2.0, max_iter: int = 100, tol: float = 1e-4):
        """
        Initialize FCM clustering.
        Args:
            n_clusters: Number of clusters.
            m: Fuzziness parameter.
            max_iter: Maximum iterations.
            tol: Tolerance for convergence.
        """
        self.n_clusters = n_clusters
        self.m = m
        self.max_iter = max_iter
        self.tol = tol
        self.centroids = None
        self.U = None

    def fit(self, X: np.ndarray) -> None:
        """
        Fit FCM to data.
        Args:
            X: Data array (samples, features).
        """
        n_samples = X.shape[0]
        U = np.random.dirichlet(np.ones(self.n_clusters), size=n_samples)
        for _ in range(self.max_iter):
            U_old = U.copy()
            um = U ** self.m
            centroids = (um.T @ X) / np.sum(um.T, axis=1, keepdims=True)
            dist = cdist(X, centroids, metric='euclidean')
            dist = np.fmax(dist, 1e-10)
            U = 1.0 / (dist ** (2/(self.m-1)))
            U = U / np.sum(U, axis=1, keepdims=True)
            if np.linalg.norm(U - U_old) < self.tol:
                break
        self.centroids = centroids
        self.U = U

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict fuzzy membership values.
        Args:
            X: Data array.
        Returns:
            Membership matrix.
        """
        dist = cdist(X, self.centroids, metric='euclidean')
        dist = np.fmax(dist, 1e-10)
        U = 1.0 / (dist ** (2/(self.m-1)))
        U = U / np.sum(U, axis=1, keepdims=True)
        return U

    def get_segmentation_map(self, X: np.ndarray) -> np.ndarray:
        """
        Generate cluster segmentation map.
        Args:
            X: Data array.
        Returns:
            Segmentation map.
        """
        U = self.predict(X)
        return np.argmax(U, axis=1)

    def get_membership_map(self, X: np.ndarray) -> np.ndarray:
        """
        Generate membership probability map.
        Args:
            X: Data array.
        Returns:
            Membership map.
        """
        return self.predict(X)
