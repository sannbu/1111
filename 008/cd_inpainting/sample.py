import argparse
import os
from tkinter import Image

import torch
import os
import numpy as np
from PIL import Image

from .cond import load_condition_and_mask
from .diffusion import Diffusion
from .unet_cond import ConditionalUNet
from .utils import load_grayscale_image, save_tensor_image, seed_all


def parse_args():
    parser = argparse.ArgumentParser(description="Sampling script for CD-Inpainting")
    parser.add_argument("--ckpt", required=True, type=str)
    parser.add_argument("--background_image", required=True, type=str)
    parser.add_argument("--label_txt", required=True, type=str)
    parser.add_argument("--out_image", required=True, type=str)
    parser.add_argument("--img_size", type=int, default=320)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--beta_start", type=float, default=1e-4)
    parser.add_argument("--beta_end", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    seed_all(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = ConditionalUNet().to(device)
    diffusion = Diffusion(
        timesteps=args.timesteps, beta_start=args.beta_start, beta_end=args.beta_end, device=device
    )

    ckpt = torch.load(args.ckpt, map_location=device)
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict)
    model.eval()

    # load background and condition
    bg = load_grayscale_image(args.background_image, args.img_size).unsqueeze(0).to(device)  # (1,1,H,W)
    cond, mask = load_condition_and_mask(args.label_txt, args.img_size)
    cond = cond.unsqueeze(0).to(device)
    mask = mask.unsqueeze(0).to(device)
    cond_img = cond[0, 0].detach().cpu().numpy()   # (H,W) numpy
    mask_img = mask[0, 0].detach().cpu().numpy()   # (H,W) numpy

    Image.fromarray((cond_img * 255).clip(0, 255).astype(np.uint8)).save("outputs/debug_cond.png")
    Image.fromarray((mask_img * 255).clip(0, 255).astype(np.uint8)).save("outputs/debug_mask.png")

    print("Saved outputs/debug_cond.png and outputs/debug_mask.png")
    with torch.no_grad():
        sample = diffusion.sample_inpaint(
            model=model,
            cond=cond,
            mask=mask,
            background=bg,
            img_size=args.img_size,
            device=device,
            timesteps=args.timesteps,
            seed=args.seed,
        )
    os.makedirs(os.path.dirname(args.out_image), exist_ok=True)
    save_tensor_image(sample, args.out_image)
    print(f"Saved generated image to {args.out_image}")


if __name__ == "__main__":
    main()
