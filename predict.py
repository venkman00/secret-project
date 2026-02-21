"""
Inference pipeline: CNN prediction → Test-Time Optimization → OpenCV undistort → zip.

Two-stage approach:
1. CNN forward pass → initial (k1, k2, k3, cx, cy) prediction
2. Test-Time Optimization (TTO): refine per-image using self-supervised geometric losses

Usage:
    python predict.py --test_dir data/test/ --checkpoint checkpoints/best_model.pth --output_dir output/
"""

import argparse
import os
import zipfile
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import TestDataset
from model import DistortionNet


def undistort_opencv(image, k1, k2, k3, cx, cy, alpha=0):
    """Apply undistortion using OpenCV (for final output quality)."""
    h, w = image.shape[:2]

    fx = fy = max(h, w)
    camera_matrix = np.array([
        [fx, 0, cx * w],
        [0, fy, cy * h],
        [0, 0, 1]
    ], dtype=np.float64)

    dist_coeffs = np.array([k1, k2, 0, 0, k3], dtype=np.float64)

    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, (w, h), alpha=alpha
    )

    undistorted = cv2.undistort(image, camera_matrix, dist_coeffs, None, new_camera_matrix)

    # Crop to ROI
    x, y, rw, rh = roi
    if rw > 0 and rh > 0:
        undistorted = undistorted[y:y+rh, x:x+rw]
        # Resize back to original dimensions
        undistorted = cv2.resize(undistorted, (w, h), interpolation=cv2.INTER_LANCZOS4)

    return undistorted


def compute_tto_loss(image_tensor, params):
    """
    Self-supervised losses for test-time optimization.
    No labels needed — uses geometric priors only.

    Args:
        image_tensor: (1, C, H, W) undistorted image
        params: (1, 5) current parameters [k1, k2, k3, cx, cy]

    Returns:
        loss: scalar tensor
    """
    img = image_tensor.squeeze(0)  # (C, H, W)
    C, H, W = img.shape

    # Convert to grayscale
    gray = 0.299 * img[0] + 0.587 * img[1] + 0.114 * img[2]  # (H, W)
    gray = gray.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)

    # 1. Edge sharpness loss: gradient magnitudes should be high
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                           dtype=torch.float32, device=img.device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                           dtype=torch.float32, device=img.device).view(1, 1, 3, 3)

    gx = F.conv2d(gray, sobel_x, padding=1)
    gy = F.conv2d(gray, sobel_y, padding=1)
    gradient_mag = torch.sqrt(gx**2 + gy**2 + 1e-6)

    # We want to maximize edge sharpness → minimize negative
    edge_loss = -gradient_mag.mean()

    # 2. Border coverage loss: minimize black border regions
    # Check border pixels — they should not be zero (black)
    border_size = max(2, min(H, W) // 20)
    borders = torch.cat([
        img[:, :border_size, :].reshape(-1),      # top
        img[:, -border_size:, :].reshape(-1),      # bottom
        img[:, :, :border_size].reshape(-1),       # left
        img[:, :, -border_size:].reshape(-1),      # right
    ])
    border_loss = (1.0 - borders.abs().clamp(0, 1)).mean()

    # 3. Smoothness regularization: params shouldn't deviate too far from CNN prediction
    # (handled externally by initialization)

    total_loss = 0.5 * edge_loss + 0.5 * border_loss
    return total_loss


def differentiable_undistort_single(image, params):
    """Differentiable undistortion for a single image during TTO."""
    B, C, H, W = image.shape

    k1, k2, k3 = params[0, 0], params[0, 1], params[0, 2]
    cx, cy = params[0, 3], params[0, 4]

    grid_y, grid_x = torch.meshgrid(
        torch.linspace(-1, 1, H, device=image.device),
        torch.linspace(-1, 1, W, device=image.device),
        indexing='ij'
    )

    cx_norm = cx * 2 - 1
    cy_norm = cy * 2 - 1

    dx = grid_x - cx_norm
    dy = grid_y - cy_norm
    r2 = dx**2 + dy**2

    radial = 1 + k1 * r2 + k2 * r2**2 + k3 * r2**3

    map_x = dx * radial + cx_norm
    map_y = dy * radial + cy_norm

    grid = torch.stack([map_x, map_y], dim=-1).unsqueeze(0)

    undistorted = F.grid_sample(image, grid, mode='bilinear',
                                padding_mode='zeros', align_corners=True)
    return undistorted


def test_time_optimize(image_tensor, init_params, n_steps=50, lr=0.001):
    """
    Refine distortion parameters per-image using self-supervised losses.

    Args:
        image_tensor: (1, C, H, W) distorted input image
        init_params: (5,) initial parameters from CNN
        n_steps: optimization steps
        lr: learning rate

    Returns:
        refined_params: (5,) optimized parameters
    """
    device = image_tensor.device

    # Create optimizable parameters initialized from CNN prediction
    params = torch.tensor(init_params, dtype=torch.float32, device=device).unsqueeze(0)
    params = params.clone().detach().requires_grad_(True)

    optimizer = torch.optim.Adam([params], lr=lr)
    init_params_tensor = torch.tensor(init_params, dtype=torch.float32, device=device)

    best_loss = float('inf')
    best_params = init_params.copy()

    for step in range(n_steps):
        optimizer.zero_grad()

        # Clamp parameters to valid ranges
        with torch.no_grad():
            params.data[:, :3].clamp_(-1, 1)
            params.data[:, 3:].clamp_(0.1, 0.9)

        # Apply undistortion
        undistorted = differentiable_undistort_single(image_tensor, params)

        # Compute self-supervised loss
        loss = compute_tto_loss(undistorted, params)

        # Regularization: stay close to CNN prediction
        reg_loss = 0.1 * F.mse_loss(params.squeeze(), init_params_tensor)
        total_loss = loss + reg_loss

        total_loss.backward()
        optimizer.step()

        if total_loss.item() < best_loss:
            best_loss = total_loss.item()
            best_params = params.detach().squeeze().cpu().numpy().copy()

    return best_params


def predict_batch(model, dataloader, device, use_tto=True, tto_steps=50):
    """Run inference on test images."""
    model.eval()
    predictions = {}

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="CNN prediction"):
            images = batch['image'].to(device)
            image_ids = batch['image_id']

            params = model(images)

            for i, img_id in enumerate(image_ids):
                predictions[img_id] = {
                    'params': params[i].cpu().numpy(),
                    'image_path': batch['image_path'][i],
                    'orig_h': batch['orig_h'][i].item(),
                    'orig_w': batch['orig_w'][i].item(),
                }

    # Test-time optimization (per-image)
    if use_tto:
        print(f"\nRunning test-time optimization ({tto_steps} steps per image)...")
        for img_id in tqdm(predictions, desc="TTO"):
            pred = predictions[img_id]

            # Load and preprocess image for TTO
            img = cv2.imread(pred['image_path'])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Resize to manageable size for TTO
            tto_size = 256
            img_small = cv2.resize(img, (tto_size, tto_size))
            img_tensor = torch.from_numpy(img_small).float().permute(2, 0, 1) / 255.0
            img_tensor = img_tensor.unsqueeze(0).to(device)

            refined = test_time_optimize(
                img_tensor, pred['params'], n_steps=tto_steps
            )
            pred['params_refined'] = refined

    return predictions


def apply_corrections(predictions, output_dir, use_tto=True):
    """Apply corrections and save output images."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for img_id, pred in tqdm(predictions.items(), desc="Applying corrections"):
        # Load original full-res image
        img = cv2.imread(pred['image_path'])

        if img is None:
            print(f"Warning: cannot read {pred['image_path']}")
            continue

        # Use refined params if available, else CNN params
        if use_tto and 'params_refined' in pred:
            params = pred['params_refined']
        else:
            params = pred['params']

        k1, k2, k3, cx, cy = params

        # Apply OpenCV undistortion (highest quality)
        corrected = undistort_opencv(img, k1, k2, k3, cx, cy, alpha=0)

        # Save as JPEG
        output_path = output_dir / f"{img_id}.jpg"
        cv2.imwrite(str(output_path), corrected, [cv2.IMWRITE_JPEG_QUALITY, 95])

    print(f"Saved {len(predictions)} corrected images to {output_dir}")


def create_submission_zip(output_dir, zip_path='submission.zip'):
    """Create zip file for submission."""
    output_dir = Path(output_dir)

    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for img_file in sorted(output_dir.glob('*.jpg')):
            zf.write(img_file, img_file.name)

    print(f"Created submission zip: {zip_path}")
    print(f"Upload to: https://bounty.autohdr.com")
    return zip_path


def main():
    parser = argparse.ArgumentParser(description='Predict corrected images')
    parser.add_argument('--test_dir', type=str, required=True, help='Directory with test images')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pth')
    parser.add_argument('--output_dir', type=str, default='output/')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--image_size', type=int, default=384)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--no_tto', action='store_true', help='Disable test-time optimization')
    parser.add_argument('--tto_steps', type=int, default=50)
    parser.add_argument('--backbone', type=str, default='efficientnet_b3')
    parser.add_argument('--zip_path', type=str, default='submission.zip')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    model = DistortionNet(backbone=args.backbone, pretrained=False)

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']} (val_loss={checkpoint['val_loss']:.4f})")

    # Create test dataset
    test_dataset = TestDataset(args.test_dir, image_size=args.image_size)
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True,
    )
    print(f"Test images: {len(test_dataset)}")

    # Predict
    use_tto = not args.no_tto
    predictions = predict_batch(model, test_loader, device,
                               use_tto=use_tto, tto_steps=args.tto_steps)

    # Apply corrections
    apply_corrections(predictions, args.output_dir, use_tto=use_tto)

    # Create submission zip
    create_submission_zip(args.output_dir, args.zip_path)


if __name__ == '__main__':
    main()
