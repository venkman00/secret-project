"""
Local implementation of competition scoring metrics.

Metrics and weights:
    - Edge Similarity: 40%   (Multi-scale Canny edge F1)
    - Line Straightness: 22% (Hough line angle distribution match)
    - Gradient Orientation: 18% (Gradient direction histogram similarity)
    - SSIM: 15%              (Structural similarity index)
    - Pixel Accuracy: 5%     (Mean absolute pixel difference → score)

Usage:
    from metrics import compute_score
    score = compute_score(corrected_image, ground_truth_image)
"""

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim


def edge_similarity(img1, img2, scales=(1.0, 0.5, 0.25)):
    """Multi-scale Canny edge F1 score."""
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) if len(img1.shape) == 3 else img1
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) if len(img2.shape) == 3 else img2

    f1_scores = []
    for scale in scales:
        if scale != 1.0:
            h, w = int(gray1.shape[0] * scale), int(gray1.shape[1] * scale)
            g1 = cv2.resize(gray1, (w, h))
            g2 = cv2.resize(gray2, (w, h))
        else:
            g1, g2 = gray1, gray2

        # Canny edges
        median1 = np.median(g1)
        edges1 = cv2.Canny(g1, int(max(0, 0.67 * median1)), int(min(255, 1.33 * median1)))

        median2 = np.median(g2)
        edges2 = cv2.Canny(g2, int(max(0, 0.67 * median2)), int(min(255, 1.33 * median2)))

        # Dilate edges slightly for tolerance
        kernel = np.ones((3, 3), np.uint8)
        edges1_dilated = cv2.dilate(edges1, kernel, iterations=1)
        edges2_dilated = cv2.dilate(edges2, kernel, iterations=1)

        # Precision: fraction of pred edges that match GT
        pred_on = edges1 > 0
        gt_on = edges2 > 0

        if pred_on.sum() == 0 and gt_on.sum() == 0:
            f1_scores.append(1.0)
            continue
        if pred_on.sum() == 0 or gt_on.sum() == 0:
            f1_scores.append(0.0)
            continue

        precision = (edges1 & edges2_dilated).sum() / max(edges1.sum(), 1)
        recall = (edges2 & edges1_dilated).sum() / max(edges2.sum(), 1)

        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0

        f1_scores.append(float(f1))

    return np.mean(f1_scores)


def line_straightness(img, reference_img=None):
    """Hough line angle distribution match.

    Measures how well the line angle distribution in img matches reference_img.
    Straighter lines = better correction.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img

    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50,
                            minLineLength=30, maxLineGap=10)

    if lines is None or len(lines) == 0:
        return 0.5  # no lines detected, neutral score

    # Extract angles
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
        angles.append(angle)
    angles = np.array(angles)

    if reference_img is not None:
        # Compare angle distributions
        gray_ref = cv2.cvtColor(reference_img, cv2.COLOR_BGR2GRAY) if len(reference_img.shape) == 3 else reference_img
        edges_ref = cv2.Canny(gray_ref, 50, 150, apertureSize=3)
        lines_ref = cv2.HoughLinesP(edges_ref, 1, np.pi / 180, threshold=50,
                                     minLineLength=30, maxLineGap=10)

        if lines_ref is None or len(lines_ref) == 0:
            return 0.5

        angles_ref = []
        for line in lines_ref:
            x1, y1, x2, y2 = line[0]
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
            angles_ref.append(angle)
        angles_ref = np.array(angles_ref)

        # Histogram comparison
        bins = np.linspace(-90, 90, 37)
        hist1, _ = np.histogram(angles, bins=bins, density=True)
        hist2, _ = np.histogram(angles_ref, bins=bins, density=True)

        # Bhattacharyya similarity
        hist1 = hist1 / (hist1.sum() + 1e-10)
        hist2 = hist2 / (hist2.sum() + 1e-10)
        bc = np.sum(np.sqrt(hist1 * hist2))
        return float(bc)
    else:
        # Self-assessment: measure how close angles are to 0, 90, or 180 degrees
        # (i.e., how "straight" the lines are in cardinal directions)
        closest = np.minimum(
            np.minimum(np.abs(angles), np.abs(angles - 90)),
            np.minimum(np.abs(angles + 90), np.abs(angles - 180))
        )
        # Score: fraction of lines close to cardinal directions
        score = np.mean(closest < 5)  # within 5 degrees
        return float(score)


def gradient_orientation_similarity(img1, img2, n_bins=36):
    """Gradient direction histogram similarity."""
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) if len(img1.shape) == 3 else img1
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) if len(img2.shape) == 3 else img2

    gray1 = gray1.astype(np.float32)
    gray2 = gray2.astype(np.float32)

    # Compute gradients
    gx1 = cv2.Sobel(gray1, cv2.CV_32F, 1, 0, ksize=3)
    gy1 = cv2.Sobel(gray1, cv2.CV_32F, 0, 1, ksize=3)
    gx2 = cv2.Sobel(gray2, cv2.CV_32F, 1, 0, ksize=3)
    gy2 = cv2.Sobel(gray2, cv2.CV_32F, 0, 1, ksize=3)

    # Gradient orientations
    orient1 = np.arctan2(gy1, gx1)  # [-pi, pi]
    orient2 = np.arctan2(gy2, gx2)
    mag1 = np.sqrt(gx1**2 + gy1**2)
    mag2 = np.sqrt(gx2**2 + gy2**2)

    # Magnitude-weighted orientation histograms
    bins = np.linspace(-np.pi, np.pi, n_bins + 1)

    hist1, _ = np.histogram(orient1, bins=bins, weights=mag1)
    hist2, _ = np.histogram(orient2, bins=bins, weights=mag2)

    # Normalize
    hist1 = hist1 / (hist1.sum() + 1e-10)
    hist2 = hist2 / (hist2.sum() + 1e-10)

    # Bhattacharyya coefficient
    bc = np.sum(np.sqrt(hist1 * hist2))
    return float(bc)


def compute_ssim(img1, img2):
    """Compute SSIM between two images."""
    # Ensure same size
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    if len(img1.shape) == 3:
        return ssim(img1, img2, channel_axis=2, data_range=255)
    else:
        return ssim(img1, img2, data_range=255)


def pixel_accuracy(img1, img2):
    """Pixel accuracy score (1 - normalized MAE)."""
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    mae = np.mean(np.abs(img1.astype(np.float32) - img2.astype(np.float32)))
    # Normalize to [0, 1]: 0 MAE = 1.0 score, 255 MAE = 0.0 score
    score = 1.0 - mae / 255.0
    return float(score)


def compute_score(corrected, ground_truth):
    """
    Compute the full competition score for a corrected image vs ground truth.

    Returns:
        float: Overall score in [0, 1]
        dict: Individual metric scores
    """
    # Ensure same size
    if corrected.shape != ground_truth.shape:
        corrected = cv2.resize(corrected, (ground_truth.shape[1], ground_truth.shape[0]))

    edge_sim = edge_similarity(corrected, ground_truth)
    line_str = line_straightness(corrected, ground_truth)
    grad_orient = gradient_orientation_similarity(corrected, ground_truth)
    ssim_score = compute_ssim(corrected, ground_truth)
    pixel_acc = pixel_accuracy(corrected, ground_truth)

    # Weighted combination
    overall = (
        0.40 * edge_sim +
        0.22 * line_str +
        0.18 * grad_orient +
        0.15 * ssim_score +
        0.05 * pixel_acc
    )

    metrics = {
        'edge_similarity': edge_sim,
        'line_straightness': line_str,
        'gradient_orientation': grad_orient,
        'ssim': ssim_score,
        'pixel_accuracy': pixel_acc,
        'overall': overall,
    }

    return overall, metrics


def evaluate_batch(corrected_dir, ground_truth_dir, max_images=None):
    """Evaluate all corrected images against ground truth."""
    from pathlib import Path

    corr_dir = Path(corrected_dir)
    gt_dir = Path(ground_truth_dir)

    img_exts = {'.jpg', '.jpeg', '.png', '.bmp'}
    corr_files = sorted([f for f in corr_dir.iterdir() if f.suffix.lower() in img_exts])

    if max_images:
        corr_files = corr_files[:max_images]

    all_scores = []
    for cf in corr_files:
        # Find matching GT
        gt_path = None
        for ext in img_exts:
            candidate = gt_dir / (cf.stem + ext)
            if candidate.exists():
                gt_path = candidate
                break

        if gt_path is None:
            continue

        corrected = cv2.imread(str(cf))
        ground_truth = cv2.imread(str(gt_path))

        if corrected is None or ground_truth is None:
            continue

        overall, metrics = compute_score(corrected, ground_truth)
        metrics['image_id'] = cf.stem
        all_scores.append(metrics)
        print(f"  {cf.stem}: {overall:.4f} "
              f"(edge={metrics['edge_similarity']:.3f}, "
              f"line={metrics['line_straightness']:.3f}, "
              f"grad={metrics['gradient_orientation']:.3f}, "
              f"ssim={metrics['ssim']:.3f}, "
              f"pixel={metrics['pixel_accuracy']:.3f})")

    if all_scores:
        avg_overall = np.mean([s['overall'] for s in all_scores])
        print(f"\nAverage score: {avg_overall:.4f}")
        return all_scores

    return []


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate corrected images')
    parser.add_argument('--corrected_dir', type=str, required=True)
    parser.add_argument('--ground_truth_dir', type=str, required=True)
    parser.add_argument('--max_images', type=int, default=None)
    args = parser.parse_args()

    evaluate_batch(args.corrected_dir, args.ground_truth_dir, args.max_images)
