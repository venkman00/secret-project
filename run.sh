#!/bin/bash
# =============================================================================
# Automatic Lens Correction Pipeline
# Run on Colab or any GPU machine.
# =============================================================================
set -e

echo "=== Phase 0: Setup ==="

# Install dependencies
pip install -q torch torchvision timm opencv-python-headless scipy kornia \
    pytorch-msssim scikit-image albumentations kaggle pandas tqdm Pillow numpy

# Download competition data (requires ~/.kaggle/kaggle.json)
if [ ! -d "data" ]; then
    echo "Downloading competition data..."
    kaggle competitions download -c automatic-lens-correction
    mkdir -p data
    unzip -q automatic-lens-correction.zip -d data/
    echo "Data downloaded and extracted to data/"
else
    echo "Data directory already exists, skipping download"
fi

# Show data structure
echo "Data structure:"
find data/ -maxdepth 3 -type d | head -20
echo "Image count:"
find data/ -name "*.jpg" -o -name "*.png" -o -name "*.jpeg" | wc -l

echo ""
echo "=== Phase 1: Extract Parameters ==="
python extract_params.py --data_dir data/ --output params.csv --workers 8 --validate 5

echo ""
echo "=== Phase 2: Train Model ==="
python train.py \
    --data_dir data/ \
    --params_csv params.csv \
    --epochs 30 \
    --batch_size 16 \
    --lr 3e-3 \
    --progressive \
    --save_dir checkpoints/

echo ""
echo "=== Phase 4: Predict Test Images ==="
# Auto-detect test directory
TEST_DIR=$(find data/ -type d -name "test*" | head -1)
if [ -z "$TEST_DIR" ]; then
    echo "Could not find test directory. Listing data/ contents:"
    ls -la data/
    echo "Set TEST_DIR manually and rerun predict.py"
    exit 1
fi

python predict.py \
    --test_dir "$TEST_DIR" \
    --checkpoint checkpoints/best_model.pth \
    --output_dir output/ \
    --tto_steps 50 \
    --zip_path submission.zip

echo ""
echo "=== Done! ==="
echo "1. Upload submission.zip to https://bounty.autohdr.com"
echo "2. Download the scoring CSV"
echo "3. Submit CSV to Kaggle"
