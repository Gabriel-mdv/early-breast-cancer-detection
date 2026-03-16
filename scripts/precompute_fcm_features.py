"""
Script to precompute and save FCM membership maps for all images in the processed dataset.
"""

import numpy as np
from pathlib import Path
from fcm.fcm import FuzzyCMeans
import os

# Settings
image_root = Path('datasets/processed/images')
out_root = Path('datasets/processed/fcm_features')
out_root.mkdir(parents=True, exist_ok=True)
n_clusters = 3  # or set as needed

# Helper to process one image file
def compute_fcm_features(image_path, n_clusters=3):
    img = np.load(image_path)
    flat = img.flatten().reshape(-1, 1)
    fcm = FuzzyCMeans(n_clusters=n_clusters)
    fcm.fit(flat)
    membership = fcm.get_membership_map(flat)  # shape: (num_pixels, n_clusters)
    membership_map = membership.reshape(img.shape + (n_clusters,))
    return membership_map

# Process all images
for class_dir in image_root.iterdir():
    if not class_dir.is_dir():
        continue
    out_class_dir = out_root / class_dir.name
    out_class_dir.mkdir(exist_ok=True)
    for img_file in class_dir.glob('*.npy'):
        out_file = out_class_dir / img_file.name
        if out_file.exists():
            continue  # skip if already computed
        membership_map = compute_fcm_features(img_file, n_clusters=n_clusters)
        np.save(out_file, membership_map)
        print(f'Saved FCM features: {out_file}')

print('FCM feature precomputation complete.')
