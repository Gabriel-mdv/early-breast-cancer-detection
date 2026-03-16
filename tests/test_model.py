
"""
Unit test for MobileFCMViTv3 model forward pass.
"""

import unittest
import torch
from models.mobilefcmvitv3_model import MobileFCMViTv3

class TestModel(unittest.TestCase):
    def test_forward(self):
        model = MobileFCMViTv3(1, 3, 2)
        img = torch.randn(1, 1, 224, 224)
        fcm_feat = torch.randn(1, 3, 224, 224)
        out = model(img, fcm_feat)
        self.assertEqual(out.shape, (1, 2))
