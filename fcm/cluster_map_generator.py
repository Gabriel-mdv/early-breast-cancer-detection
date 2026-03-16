"""
Cluster map generator for FCM outputs.
"""

import numpy as np
from typing import Any
from .fuzzy_c_means import FuzzyCMeans

class ClusterMapGenerator:
    def __init__(self, n_clusters: int = 3):
        self.fcm = FuzzyCMeans(n_clusters=n_clusters)

    def generate(self, X: np.ndarray) -> np.ndarray:
        """
        Generate cluster segmentation map.
        Args:
            X: Data array.
        Returns:
            Segmentation map.
        """
        self.fcm.fit(X)
        return self.fcm.get_segmentation_map(X)
