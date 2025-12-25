import os
import random
from typing import Optional

import numpy as np
from PIL import Image
import torch

from .cond import labels_to_condition_and_mask
from .diffusion import Diffusion
from .unet_cond import ConditionalUNet


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_tensor_image(tensor: torch.Tensor, path: str):
    """Save a tensor in [-1,1] to disk as grayscale PNG."""
    tensor = tensor.detach().cpu().clamp(-1.0, 1.0)
    tensor = (tensor + 1.0) / 2.0  # [0,1]
    array = (tensor.squeeze(0).squeeze(0).numpy() * 255.0).astype(np.uint8)
    img = Image.fromarray(array, mode="L")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img.save(path)


def load_grayscale_image(path: str, img_size: int) -> torch.Tensor:
    img = Image.open(path).convert("L")
    img = img.resize((img_size, img_size), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = arr * 2.0 - 1.0
    return torch.from_numpy(arr).unsqueeze(0)


def smoke_test(output_path: str = "cd_inpainting/smoke_test.png", img_size: int = 64):
    """Run a minimal smoke test: one forward diffusion step, UNet forward, one reverse step."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed_all(123)
    model = ConditionalUNet(window_size=8).to(device)
    diffusion = Diffusion(timesteps=10, device=device)

    # dummy data
    x0 = torch.randn(1, 1, img_size, img_size, device=device).clamp(-1, 1)
    labels = [(0, 0.5, 0.5, 0.25, 0.25)]
    cond, mask = labels_to_condition_and_mask(labels, img_size)
    cond = cond.unsqueeze(0).to(device)
    mask = mask.unsqueeze(0).to(device)
    t = torch.tensor([5], device=device, dtype=torch.long)
    noise = torch.randn_like(x0)
    x_t = diffusion.q_sample(x0, t, noise)

    # UNet forward
    _ = model(x_t, cond, t)

    # one reverse sampling step with inpainting enforcement
    sample, _ = diffusion.p_sample_inpaint(model, x_t, cond, mask, x0, t)
    save_tensor_image(sample.cpu(), output_path)
    return output_path


if __name__ == "__main__":
    print("Running smoke test...")
    out = smoke_test()
    print(f"Smoke test image saved to {out}")
