# Conditional Diffusion Inpainting for Infrared Target Generation

This repository implements the CD-Inpainting method described in “Infrared Target Generation based on Conditional Diffusion Inpainting (CD-Inpainting), SPIE CSTA 2024, Bo Dong et al.” with literal adherence to the provided specification: conditional DDPM, annotation-to-image conditioning, multi-scale UNet with LocalAttention, and inpainting sampling that preserves background outside annotated boxes.

## Directory Layout
- `configs.py` – simple config dataclasses.
- `cond.py` – YOLO label parsing, condition image `d_I` and union mask `M`.
- `dataset.py` – dataset loader returning grayscale images, condition, and mask.
- `unet_cond.py` – image-conditioned UNet architecture per Table 1 with multi-scale condition concat, timestep embedding, and LocalAttention.
- `diffusion.py` – DDPM forward/reverse utilities and inpainting-aware sampling.
- `train.py` – training loop (epsilon prediction MSE).
- `sample.py` – sampling/inpainting script.
- `utils.py` – seeding, IO helpers, smoke test function.
- `paper_alignment.md` – notes on resolving table ambiguities.

## Setup
Install dependencies (PyTorch, Pillow, NumPy):

```bash
pip install torch torchvision pillow numpy
```

## Training

```bash
python -m cd_inpainting.train \
  --images_dir data/train/images \
  --labels_dir data/train/labels \
  --out_dir runs/cd_inpainting \
  --img_size 320 \
  --timesteps 1000 \
  --beta_start 1e-4 --beta_end 2e-2 \
  --batch_size 4 --lr 2e-4 --epochs 100 --num_workers 4 --seed 42
```

Checkpoints are saved to `--out_dir/ckpt_epoch_*.pt`.

## Sampling / Inpainting

```bash
python -m cd_inpainting.sample \
  --ckpt runs/cd_inpainting/ckpt_epoch_100.pt \
  --background_image path/to/background.png \
  --label_txt path/to/labels.txt \
  --out_image outputs/generated.png \
  --img_size 320 --timesteps 1000 --beta_start 1e-4 --beta_end 2e-2 --seed 0
```

Sampling enforces background outside the union mask at every reverse step via the specified inpainting rule.

## Smoke Test

```bash
python -m cd_inpainting.utils
```

This runs a minimal forward diffusion step, UNet forward, one reverse sampling step, and writes `cd_inpainting/smoke_test.png` to verify shapes.
