
"""
Unit test for UltrasoundDataset.
"""

import unittest
from datasets.ultrasound_dataset import UltrasoundDataset

class TestUltrasoundDataset(unittest.TestCase):
    def test_len(self):
        ds = UltrasoundDataset(['img1.png', 'img2.png'], [0, 1])
        self.assertEqual(len(ds), 2)
