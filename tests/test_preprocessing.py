
"""
Unit test for UltrasoundPreprocessingPipeline.
"""

import unittest
from preprocessing.preprocessing_pipeline import UltrasoundPreprocessingPipeline
from PIL import Image
import numpy as np

class TestPreprocessing(unittest.TestCase):
    def test_preprocess(self):
        pipeline = UltrasoundPreprocessingPipeline()
        img = Image.fromarray(np.random.randint(0, 255, (224, 224), dtype=np.uint8))
        out = pipeline.preprocess(img)
        self.assertEqual(out.shape[-2:], (224, 224))
