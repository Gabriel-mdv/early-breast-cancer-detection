
"""
Unit test for FuzzyCMeans clustering.
"""

import unittest
import numpy as np
from clustering.fuzzy_c_means import FuzzyCMeans

class TestFCM(unittest.TestCase):
    def test_fit_predict(self):
        X = np.random.rand(10, 2)
        fcm = FuzzyCMeans(n_clusters=2)
        fcm.fit(X)
        U = fcm.predict(X)
        self.assertEqual(U.shape, (10, 2))
