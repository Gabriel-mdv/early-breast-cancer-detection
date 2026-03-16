"""
Membership map generator for FCM outputs.
"""

import numpy as np
from typing import Any
from .fuzzy_c_means import FuzzyCMeans

class MembershipMapGenerator:
    def __init__(self, n_clusters: int = 3):
        self.fcm = FuzzyCMeans(n_clusters=n_clusters)

    def generate(self, X: np.ndarray) -> np.ndarray:
        """
        Generate membership probability map.
        Args:
            X: Data array.
        Returns:
            Membership map.
        """
        self.fcm.fit(X)
        return self.fcm.get_membership_map(X)
