"""
PyTorch Dataset for (distorted_image, distortion_params) pairs.

Supports progressive resizing and appropriate augmentations
(color only — no geometric augments that would change the distortion).
"""

import csv
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset


class DistortionDataset(Dataset):
    """Dataset that loads distorted images and their extracted distortion parameters."""

    def __init__(self, image_dir, params_csv, image_size=224, augment=True,
                 corrected_dir=None):
        """
        Args:
            image_dir: Directory containing distorted images
            params_csv: CSV file with columns [image_id, k1, k2, k3, cx, cy]
            image_size: Target image size (square)
            augment: Whether to apply augmentations
            corrected_dir: Optional directory with corrected images (for pixel loss)
        """
        self.image_dir = Path(image_dir)
        self.corrected_dir = Path(corrected_dir) if corrected_dir else None
        self.image_size = image_size
        self.augment = augment

        # Load params
        self.samples = []
        with open(params_csv, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                image_id = row['image_id']
                params = np.array([
                    float(row['k1']),
                    float(row['k2']),
                    float(row['k3']),
                    float(row['cx']),
                    float(row['cy']),
                ], dtype=np.float32)
                self.samples.append((image_id, params))

        # Find actual image files (handle different extensions)
        self._image_paths = {}
        img_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        for f in self.image_dir.iterdir():
            if f.suffix.lower() in img_exts:
                self._image_paths[f.stem] = f

        # Filter to only samples we have images for
        valid = [(img_id, p) for img_id, p in self.samples if img_id in self._image_paths]
        if len(valid) < len(self.samples):
            print(f"Warning: {len(self.samples) - len(valid)} samples missing image files")
        self.samples = valid

        # Build augmentation pipeline (color only — NO geometric transforms)
        if self.augment:
            self.transform = A.Compose([
                A.Resize(image_size, image_size),
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05, p=0.5),
                A.GaussNoise(std_range=(0.01, 0.03), p=0.3),
                A.ImageCompression(quality_range=(70, 95), p=0.3),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])
        else:
            self.transform = A.Compose([
                A.Resize(image_size, image_size),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])

        # Corrected image transform (no augmentation)
        self.target_transform = A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_id, params = self.samples[idx]

        # Load distorted image
        img_path = self._image_paths[image_id]
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        transformed = self.transform(image=img)
        img_tensor = transformed['image']

        result = {
            'image': img_tensor,
            'params': torch.from_numpy(params),
            'image_id': image_id,
        }

        # Optionally load corrected image for pixel loss
        if self.corrected_dir is not None:
            corr_path = None
            for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                candidate = self.corrected_dir / (image_id + ext)
                if candidate.exists():
                    corr_path = candidate
                    break

            if corr_path is not None:
                corr_img = cv2.imread(str(corr_path))
                corr_img = cv2.cvtColor(corr_img, cv2.COLOR_BGR2RGB)
                corr_transformed = self.target_transform(image=corr_img)
                result['corrected'] = corr_transformed['image']

        return result

    def update_image_size(self, new_size):
        """Update image size for progressive resizing."""
        self.image_size = new_size
        # Rebuild transforms
        if self.augment:
            self.transform = A.Compose([
                A.Resize(new_size, new_size),
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05, p=0.5),
                A.GaussNoise(std_range=(0.01, 0.03), p=0.3),
                A.ImageCompression(quality_range=(70, 95), p=0.3),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])
        else:
            self.transform = A.Compose([
                A.Resize(new_size, new_size),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])
        self.target_transform = A.Compose([
            A.Resize(new_size, new_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])


class TestDataset(Dataset):
    """Dataset for test images (no labels)."""

    def __init__(self, image_dir, image_size=384):
        self.image_dir = Path(image_dir)
        self.image_size = image_size

        img_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        self.image_files = sorted([
            f for f in self.image_dir.iterdir()
            if f.suffix.lower() in img_exts
        ])

        self.transform = A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Store original size for full-res undistortion later
        orig_h, orig_w = img.shape[:2]

        transformed = self.transform(image=img)

        return {
            'image': transformed['image'],
            'image_id': img_path.stem,
            'image_path': str(img_path),
            'orig_h': orig_h,
            'orig_w': orig_w,
        }
