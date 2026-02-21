"""
Training loop with dual loss, progressive resizing, and one-cycle LR.

Usage:
    python train.py --data_dir data/ --params_csv params.csv --epochs 30 --batch_size 16
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from dataset import DistortionDataset
from model import DistortionNet, DistortionLoss


def find_dirs(data_dir):
    """Find distorted and corrected directories."""
    data_dir = Path(data_dir)
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
            return d, c

    # Fallback: auto-detect from subdirectories with matching files
    img_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    subdirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])
    for i, d1 in enumerate(subdirs):
        for d2 in subdirs[i+1:]:
            files1 = {f.stem for f in d1.glob('*') if f.suffix.lower() in img_exts}
            files2 = {f.stem for f in d2.glob('*') if f.suffix.lower() in img_exts}
            if len(files1 & files2) > 10:
                return d1, d2

    raise FileNotFoundError(f"Cannot find training directories in {data_dir}")


def train_one_epoch(model, loader, criterion, optimizer, scheduler, device, epoch):
    model.train()
    total_loss = 0
    total_param_loss = 0
    total_pixel_loss = 0
    n_batches = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch}")
    for batch in pbar:
        images = batch['image'].to(device)
        params = batch['params'].to(device)
        corrected = batch.get('corrected')
        if corrected is not None:
            corrected = corrected.to(device)

        pred_params = model(images)
        loss, loss_param, loss_pixel = criterion(
            pred_params, params, images, corrected
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        total_param_loss += loss_param.item()
        total_pixel_loss += loss_pixel.item()
        n_batches += 1

        pbar.set_postfix({
            'loss': f'{total_loss/n_batches:.4f}',
            'param': f'{total_param_loss/n_batches:.4f}',
            'pixel': f'{total_pixel_loss/n_batches:.4f}',
            'lr': f'{scheduler.get_last_lr()[0]:.6f}',
        })

    return total_loss / n_batches


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    total_param_loss = 0
    param_errors = []
    n_batches = 0

    for batch in loader:
        images = batch['image'].to(device)
        params = batch['params'].to(device)
        corrected = batch.get('corrected')
        if corrected is not None:
            corrected = corrected.to(device)

        pred_params = model(images)
        loss, loss_param, loss_pixel = criterion(
            pred_params, params, images, corrected
        )

        total_loss += loss.item()
        total_param_loss += loss_param.item()
        n_batches += 1

        # Per-parameter errors
        errors = (pred_params - params).abs().cpu().numpy()
        param_errors.append(errors)

    param_errors = np.concatenate(param_errors, axis=0)
    mean_errors = param_errors.mean(axis=0)

    return {
        'loss': total_loss / n_batches,
        'param_loss': total_param_loss / n_batches,
        'k1_err': mean_errors[0],
        'k2_err': mean_errors[1],
        'k3_err': mean_errors[2],
        'cx_err': mean_errors[3],
        'cy_err': mean_errors[4],
    }


def main():
    parser = argparse.ArgumentParser(description='Train distortion prediction model')
    parser.add_argument('--data_dir', type=str, default='data/')
    parser.add_argument('--params_csv', type=str, default='params.csv')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=3e-3)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--param_weight', type=float, default=1.0)
    parser.add_argument('--pixel_weight', type=float, default=0.5)
    parser.add_argument('--backbone', type=str, default='efficientnet_b3')
    parser.add_argument('--save_dir', type=str, default='checkpoints/')
    parser.add_argument('--val_split', type=float, default=0.2)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--progressive', action='store_true', default=True,
                        help='Use progressive resizing (224->384->512)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    os.makedirs(args.save_dir, exist_ok=True)

    # Find data directories
    dist_dir, corr_dir = find_dirs(args.data_dir)
    print(f"Distorted: {dist_dir}")
    print(f"Corrected: {corr_dir}")

    # Progressive resizing schedule
    if args.progressive:
        size_schedule = {
            0: 224,
            args.epochs // 3: 384,
            2 * args.epochs // 3: 512,
        }
    else:
        size_schedule = {0: 384}

    initial_size = size_schedule[0]

    # Create dataset
    full_dataset = DistortionDataset(
        image_dir=dist_dir,
        params_csv=args.params_csv,
        image_size=initial_size,
        augment=True,
        corrected_dir=corr_dir,
    )
    print(f"Total samples: {len(full_dataset)}")

    # Train/val split
    n_val = int(len(full_dataset) * args.val_split)
    n_train = len(full_dataset) - n_val
    train_dataset, val_dataset = random_split(
        full_dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )

    # Wrap val subset to disable augmentation
    val_dataset_wrapper = DistortionDataset(
        image_dir=dist_dir,
        params_csv=args.params_csv,
        image_size=initial_size,
        augment=False,
        corrected_dir=corr_dir,
    )
    # Use same indices as validation split
    val_indices = val_dataset.indices
    val_dataset_wrapper.samples = [val_dataset_wrapper.samples[i] for i in val_indices]

    print(f"Train: {n_train}, Val: {n_val}")

    # Model
    model = DistortionNet(backbone=args.backbone, pretrained=True).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Loss
    criterion = DistortionLoss(
        param_weight=args.param_weight,
        pixel_weight=args.pixel_weight,
    )

    # Optimizer & scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    steps_per_epoch = len(train_dataset) // args.batch_size + 1
    total_steps = steps_per_epoch * args.epochs
    scheduler = OneCycleLR(
        optimizer, max_lr=args.lr, total_steps=total_steps,
        pct_start=0.3, div_factor=25, final_div_factor=1000,
    )

    best_val_loss = float('inf')
    history = []

    for epoch in range(1, args.epochs + 1):
        # Check if we need to resize
        if epoch - 1 in size_schedule:
            new_size = size_schedule[epoch - 1]
            print(f"\n>>> Progressive resize: {new_size}x{new_size}")
            full_dataset.update_image_size(new_size)
            val_dataset_wrapper.update_image_size(new_size)

        # Create dataloaders (recreated if size changed)
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True, drop_last=True,
        )
        val_loader = DataLoader(
            val_dataset_wrapper, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True,
        )

        # Train
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scheduler, device, epoch)

        # Validate
        val_metrics = validate(model, val_loader, criterion, device)

        print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_metrics['loss']:.4f}, "
              f"k1_err={val_metrics['k1_err']:.4f}, k2_err={val_metrics['k2_err']:.4f}, "
              f"k3_err={val_metrics['k3_err']:.4f}, cx_err={val_metrics['cx_err']:.4f}, "
              f"cy_err={val_metrics['cy_err']:.4f}")

        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            **val_metrics,
        })

        # Save best model
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
                'args': vars(args),
            }, os.path.join(args.save_dir, 'best_model.pth'))
            print(f"  Saved best model (val_loss={best_val_loss:.4f})")

        # Save latest
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_metrics['loss'],
            'args': vars(args),
        }, os.path.join(args.save_dir, 'latest_model.pth'))

    # Save training history
    with open(os.path.join(args.save_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\nTraining complete. Best val_loss: {best_val_loss:.4f}")
    print(f"Model saved to {args.save_dir}/best_model.pth")


if __name__ == '__main__':
    main()
