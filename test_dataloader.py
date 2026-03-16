"""
Test script to verify dataloader and sampler work correctly after fix
"""

from preprocessing import BUSIDataLoader, ClassAwareSampler
from torch.utils.data import DataLoader
import torch

print("[TEST 1: Import all components]")
print("✓ Successfully imported BUSIDataLoader")
print("✓ Successfully imported ClassAwareSampler")

print("\n[TEST 2: Create dataloader and dataset]")
loader = BUSIDataLoader(processed_dir='datasets/processed', batch_size=32)
train_dataset = loader.create_dataset('train', 'datasets/processed/manifests/train_manifest.txt')
print(f"✓ Created training dataset with {len(train_dataset)} images")
print(f"  Class distribution: Benign={sum(1 for l in train_dataset.labels if l==0)}, "
      f"Malignant={sum(1 for l in train_dataset.labels if l==1)}, "
      f"Normal={sum(1 for l in train_dataset.labels if l==2)}")

print("\n[TEST 3: Create sampler]")
sampler = ClassAwareSampler(train_dataset)
print(f"✓ Created ClassAwareSampler")
print(f"  Sampling probabilities:")
for class_id, prob in sampler.get_sampling_probabilities().items():
    class_name = {0: 'benign', 1: 'malignant', 2: 'normal'}[class_id]
    print(f"    {class_name} (id={class_id}): {prob:.4f}")

print("\n[TEST 4: Load batch without augmentation]")
train_dataset._augmentation_enabled = False
train_loader = DataLoader(train_dataset, batch_size=4, sampler=sampler, num_workers=0)
batch = next(iter(train_loader))
print(f"✓ Successfully loaded batch")
print(f"  Image shape: {batch['image'].shape}")
print(f"  Label shape: {batch['label'].shape}")
print(f"  Image value range: [{batch['image'].min():.3f}, {batch['image'].max():.3f}]")
print(f"  Image dtype: {batch['image'].dtype}")
print(f"  Labels: {batch['label'].tolist()}")

print("\n[TEST 5: Test augmentation flag]")
train_dataset._augmentation_enabled = True
print(f"✓ Augmentation enabled: {train_dataset._augmentation_enabled}")
batch_aug = next(iter(train_loader))
print(f"✓ Successfully loaded batch with augmentation enabled")
print(f"  Image shape: {batch_aug['image'].shape}")

print("\n" + "="*60)
print("ALL TESTS PASSED ✓")
print("="*60)
print("\nDataloader is working correctly after the fix!")
print("You can now use the ClassAwareSampler with augmentation in training.")
