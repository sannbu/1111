import os
from typing import List, Tuple

import torch


CLASS_GRAY_VALUES = [115, 130, 145, 160, 175, 190, 205, 220, 235, 250]


def read_yolo_labels(label_path: str) -> List[Tuple[int, float, float, float, float]]:
    """Read YOLO txt labels. Returns list of (cls, xc, yc, w, h)."""
    labels: List[Tuple[int, float, float, float, float]] = []
    if not os.path.exists(label_path):
        return labels
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            parts = stripped.split()
            if len(parts) != 5:
                continue
            cls, xc, yc, w, h = parts
            labels.append((int(cls), float(xc), float(yc), float(w), float(h)))
    return labels


def labels_to_condition_and_mask(
    labels: List[Tuple[int, float, float, float, float]], img_size: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert YOLO labels to condition image d_I in [0,1] and mask M in {0,1}.

    Args:
        labels: list of (cls, xc, yc, w, h) normalized to [0,1]
        img_size: spatial size (assumed square)
    Returns:
        condition: (1, H, W) float tensor in [0,1]
        mask: (1, H, W) float tensor 0/1
    """
    condition = torch.zeros((1, img_size, img_size), dtype=torch.float32)
    mask = torch.zeros((1, img_size, img_size), dtype=torch.float32)
    for cls, xc, yc, w, h in labels:
        gray = CLASS_GRAY_VALUES[cls] if 0 <= cls < len(CLASS_GRAY_VALUES) else CLASS_GRAY_VALUES[0]
        x_center = xc * img_size
        y_center = yc * img_size
        box_w = w * img_size
        box_h = h * img_size
        x0 = max(int(round(x_center - box_w / 2)), 0)
        y0 = max(int(round(y_center - box_h / 2)), 0)
        x1 = min(int(round(x_center + box_w / 2)), img_size - 1)
        y1 = min(int(round(y_center + box_h / 2)), img_size - 1)
        if x1 <= x0 or y1 <= y0:
            continue
        condition[0, y0:y1, x0:x1] = gray / 255.0
        mask[0, y0:y1, x0:x1] = 1.0
    return condition, mask


def load_condition_and_mask(label_path: str, img_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
    labels = read_yolo_labels(label_path)
    return labels_to_condition_and_mask(labels, img_size)
