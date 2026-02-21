"""
EfficientNet-B3 backbone + regression head for distortion parameter prediction.

Includes differentiable undistortion for end-to-end pixel-level training.

Output: [k1, k2, k3, cx, cy] — 5 scalars describing barrel distortion.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class DistortionNet(nn.Module):
    """Predicts distortion parameters from a single image."""

    def __init__(self, backbone='efficientnet_b3', pretrained=True):
        super().__init__()

        self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
        feat_dim = self.backbone.num_features  # 1536 for efficientnet_b3

        self.head = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 5),
        )

        # Initialize head for near-zero distortion output
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)
        # Set cx, cy bias to 0.5 (image center)
        with torch.no_grad():
            self.head[-1].bias[3] = 0.5
            self.head[-1].bias[4] = 0.5

    def forward(self, x):
        features = self.backbone(x)
        params = self.head(features)

        # Constrain outputs to valid ranges
        k = torch.tanh(params[:, :3])          # k1, k2, k3 in [-1, 1]
        center = torch.sigmoid(params[:, 3:])  # cx, cy in [0, 1]

        return torch.cat([k, center], dim=1)


def differentiable_undistort(image, params, output_size=None):
    """
    Apply undistortion using predicted parameters, differentiable via grid_sample.

    Args:
        image: (B, C, H, W) input distorted image
        params: (B, 5) predicted [k1, k2, k3, cx, cy]
        output_size: optional (H, W) for output, defaults to input size

    Returns:
        (B, C, H, W) undistorted image
    """
    B, C, H, W = image.shape
    if output_size is None:
        out_H, out_W = H, W
    else:
        out_H, out_W = output_size

    k1 = params[:, 0:1]  # (B, 1)
    k2 = params[:, 1:2]
    k3 = params[:, 2:3]
    cx = params[:, 3:4]
    cy = params[:, 4:5]

    # Create normalized coordinate grid [-1, 1]
    grid_y, grid_x = torch.meshgrid(
        torch.linspace(-1, 1, out_H, device=image.device),
        torch.linspace(-1, 1, out_W, device=image.device),
        indexing='ij'
    )
    grid_x = grid_x.unsqueeze(0).expand(B, -1, -1)  # (B, H, W)
    grid_y = grid_y.unsqueeze(0).expand(B, -1, -1)

    # Shift to distortion center (convert cx, cy from [0,1] to [-1,1])
    cx_norm = (cx * 2 - 1).unsqueeze(-1)  # (B, 1, 1)
    cy_norm = (cy * 2 - 1).unsqueeze(-1)

    dx = grid_x - cx_norm
    dy = grid_y - cy_norm

    # Radial distance squared
    r2 = dx ** 2 + dy ** 2

    # Radial distortion factor
    k1 = k1.unsqueeze(-1)  # (B, 1, 1)
    k2 = k2.unsqueeze(-1)
    k3 = k3.unsqueeze(-1)

    radial = 1 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3

    # Apply distortion (map from undistorted to distorted coordinates)
    map_x = dx * radial + cx_norm
    map_y = dy * radial + cy_norm

    # Stack into grid for grid_sample
    grid = torch.stack([map_x, map_y], dim=-1)  # (B, H, W, 2)

    # Sample from input image
    undistorted = F.grid_sample(
        image, grid, mode='bilinear', padding_mode='zeros', align_corners=True
    )

    return undistorted


class DistortionLoss(nn.Module):
    """Combined loss: parameter MSE + pixel reprojection loss."""

    def __init__(self, param_weight=1.0, pixel_weight=1.0):
        super().__init__()
        self.param_weight = param_weight
        self.pixel_weight = pixel_weight
        self.param_loss = nn.MSELoss()
        self.pixel_loss = nn.L1Loss()

    def forward(self, pred_params, gt_params, distorted_image=None, corrected_image=None):
        """
        Args:
            pred_params: (B, 5) predicted parameters
            gt_params: (B, 5) ground truth parameters
            distorted_image: (B, C, H, W) distorted input (for pixel loss)
            corrected_image: (B, C, H, W) ground truth corrected (for pixel loss)
        """
        # Parameter MSE loss
        loss_param = self.param_loss(pred_params, gt_params)
        total_loss = self.param_weight * loss_param

        loss_pixel = torch.tensor(0.0, device=pred_params.device)

        # Pixel reprojection loss (if corrected images available)
        if distorted_image is not None and corrected_image is not None:
            pred_corrected = differentiable_undistort(distorted_image, pred_params)
            loss_pixel = self.pixel_loss(pred_corrected, corrected_image)
            total_loss = total_loss + self.pixel_weight * loss_pixel

        return total_loss, loss_param, loss_pixel
