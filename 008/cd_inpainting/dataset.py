import glob
import os
from typing import List, Tuple

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from .cond import load_condition_and_mask


def list_image_files(images_dir: str) -> List[str]:
    exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff")
    files: List[str] = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(images_dir, ext)))
    files = sorted(list(set(files)))
    return files


class InfraredDataset(Dataset):
    """Dataset returning image x0 in [-1,1], condition d_I in [0,1], and mask."""

    def __init__(self, images_dir: str, labels_dir: str, img_size: int = 320):
        super().__init__()
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.img_size = img_size
        self.image_files = list_image_files(images_dir)
        if len(self.image_files) == 0:
            raise ValueError(f"No images found in {images_dir}")

    def __len__(self) -> int:
        return len(self.image_files)

    def _load_image(self, path: str) -> torch.Tensor:
        img = Image.open(path).convert("L")
        img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = arr * 2.0 - 1.0  # normalize to [-1,1]
        tensor = torch.from_numpy(arr).unsqueeze(0)  # (1,H,W)
        return tensor

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        img_path = self.image_files[idx]
        base = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(self.labels_dir, f"{base}.txt")
        x0 = self._load_image(img_path)
        condition, mask = load_condition_and_mask(label_path, self.img_size)
        return x0, condition, mask
