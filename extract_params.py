"""
Phase 1: Extract ground-truth distortion parameters from training pairs.

For each (distorted, corrected) pair, find k1, k2, k3, cx, cy that minimize
pixel MSE between undistort(distorted, params) and corrected.

Usage:
    python extract_params.py --data_dir data/ --output params.csv --workers 8
"""

import argparse
import csv
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import minimize
from tqdm import tqdm


def undistort_image(img, k1, k2, k3, cx, cy):
    """Apply undistortion with given parameters using OpenCV."""
    h, w = img.shape[:2]

    # Camera matrix with given center
    fx = fy = max(h, w)  # focal length approximation
    camera_matrix = np.array([
        [fx, 0, cx * w],
        [0, fy, cy * h],
        [0, 0, 1]
    ], dtype=np.float64)

    dist_coeffs = np.array([k1, k2, 0, 0, k3], dtype=np.float64)

    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, (w, h), alpha=0
    )

    undistorted = cv2.undistort(img, camera_matrix, dist_coeffs, None, new_camera_matrix)

    # Crop to ROI to remove black borders
    x, y, rw, rh = roi
    if rw > 0 and rh > 0:
        undistorted = undistorted[y:y+rh, x:x+rw]
        # Resize back to original dimensions
        undistorted = cv2.resize(undistorted, (w, h), interpolation=cv2.INTER_LINEAR)

    return undistorted


def objective(params, distorted, corrected):
    """MSE between undistorted image and ground truth."""
    k1, k2, k3, cx, cy = params
    try:
        undist = undistort_image(distorted, k1, k2, k3, cx, cy)
        mse = np.mean((undist.astype(np.float32) - corrected.astype(np.float32)) ** 2)
        return mse
    except Exception:
        return 1e10


def extract_single(distorted_path, corrected_path, size=256):
    """Extract distortion parameters for a single image pair."""
    dist_img = cv2.imread(str(distorted_path))
    corr_img = cv2.imread(str(corrected_path))

    if dist_img is None or corr_img is None:
        return None

    # Downscale for speed
    dist_small = cv2.resize(dist_img, (size, size), interpolation=cv2.INTER_AREA)
    corr_small = cv2.resize(corr_img, (size, size), interpolation=cv2.INTER_AREA)

    # Initial guess
    x0 = np.array([0.0, 0.0, 0.0, 0.5, 0.5])

    # Bounds
    bounds = [
        (-1.0, 1.0),   # k1
        (-1.0, 1.0),   # k2
        (-1.0, 1.0),   # k3
        (0.3, 0.7),    # cx
        (0.3, 0.7),    # cy
    ]

    result = minimize(
        objective,
        x0,
        args=(dist_small, corr_small),
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 200, 'ftol': 1e-8}
    )

    return result.x


def find_training_pairs(data_dir):
    """Discover training image pairs from data directory.

    Tries multiple common directory structures:
    1. data/train/distorted/ + data/train/corrected/
    2. data/distorted/ + data/corrected/
    3. data/train_input/ + data/train_target/
    4. Any two subdirectories with matching filenames
    """
    data_dir = Path(data_dir)

    # Try common patterns
    patterns = [
        ('train/distorted', 'train/corrected'),
        ('train/input', 'train/target'),
        ('train_distorted', 'train_corrected'),
        ('distorted', 'corrected'),
        ('input', 'target'),
        ('train_input', 'train_target'),
        ('train/input', 'train/ground_truth'),
    ]

    for dist_dir, corr_dir in patterns:
        d = data_dir / dist_dir
        c = data_dir / corr_dir
        if d.exists() and c.exists():
            print(f"Found training pairs: {d} <-> {c}")
            return d, c

    # Fallback: look for any two subdirectories
    subdirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])
    if len(subdirs) >= 2:
        # Check if they have matching files
        for i, d1 in enumerate(subdirs):
            for d2 in subdirs[i+1:]:
                files1 = {f.stem for f in d1.glob('*') if f.suffix.lower() in ('.jpg', '.jpeg', '.png')}
                files2 = {f.stem for f in d2.glob('*') if f.suffix.lower() in ('.jpg', '.jpeg', '.png')}
                overlap = files1 & files2
                if len(overlap) > 10:  # significant overlap
                    print(f"Found matching pairs: {d1} <-> {d2} ({len(overlap)} images)")
                    return d1, d2

    # Last resort: print structure and fail
    print("Could not auto-detect directory structure. Found:")
    for item in sorted(data_dir.rglob('*'))[:50]:
        print(f"  {item.relative_to(data_dir)}")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description='Extract distortion parameters from training pairs')
    parser.add_argument('--data_dir', type=str, default='data/', help='Data directory')
    parser.add_argument('--output', type=str, default='params.csv', help='Output CSV path')
    parser.add_argument('--workers', type=int, default=8, help='Number of parallel workers')
    parser.add_argument('--size', type=int, default=256, help='Downscale size for optimization')
    parser.add_argument('--validate', type=int, default=5, help='Number of images to validate on (0 to skip)')
    args = parser.parse_args()

    dist_dir, corr_dir = find_training_pairs(args.data_dir)

    # Get image extensions
    img_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    dist_files = sorted([f for f in dist_dir.iterdir() if f.suffix.lower() in img_exts])

    print(f"Found {len(dist_files)} distorted images")

    # Match pairs
    pairs = []
    for df in dist_files:
        # Try matching by stem across extensions
        for ext in img_exts:
            cf = corr_dir / (df.stem + ext)
            if cf.exists():
                pairs.append((df, cf))
                break

    print(f"Matched {len(pairs)} image pairs")

    if len(pairs) == 0:
        print("No matching pairs found!")
        sys.exit(1)

    # Extract parameters
    results = []

    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {}
            for dist_path, corr_path in pairs:
                f = executor.submit(extract_single, dist_path, corr_path, args.size)
                futures[f] = dist_path.stem

            for f in tqdm(as_completed(futures), total=len(futures), desc="Extracting params"):
                image_id = futures[f]
                try:
                    params = f.result()
                    if params is not None:
                        results.append((image_id, *params))
                except Exception as e:
                    print(f"Error processing {image_id}: {e}")
    else:
        for dist_path, corr_path in tqdm(pairs, desc="Extracting params"):
            params = extract_single(dist_path, corr_path, args.size)
            if params is not None:
                results.append((dist_path.stem, *params))

    # Save to CSV
    with open(args.output, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['image_id', 'k1', 'k2', 'k3', 'cx', 'cy'])
        for row in results:
            writer.writerow(row)

    print(f"Saved {len(results)} parameter sets to {args.output}")

    # Validation: apply extracted params to full-res images and compute PSNR
    if args.validate > 0 and len(pairs) > 0:
        print(f"\nValidating on {min(args.validate, len(results))} images...")
        param_dict = {r[0]: r[1:] for r in results}

        psnrs = []
        for dist_path, corr_path in pairs[:args.validate]:
            if dist_path.stem not in param_dict:
                continue
            k1, k2, k3, cx, cy = param_dict[dist_path.stem]

            dist_img = cv2.imread(str(dist_path))
            corr_img = cv2.imread(str(corr_path))

            undist = undistort_image(dist_img, k1, k2, k3, cx, cy)

            mse = np.mean((undist.astype(np.float32) - corr_img.astype(np.float32)) ** 2)
            if mse > 0:
                psnr = 10 * np.log10(255**2 / mse)
            else:
                psnr = float('inf')
            psnrs.append(psnr)
            print(f"  {dist_path.stem}: PSNR = {psnr:.2f} dB")

        if psnrs:
            print(f"  Average PSNR: {np.mean(psnrs):.2f} dB (target: >30 dB)")
            if np.mean(psnrs) < 25:
                print("  WARNING: Low PSNR. Consider increasing --size or using SIFT matching.")


if __name__ == '__main__':
    main()
