import argparse
import os

import torch
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_

from .dataset import InfraredDataset
from .diffusion import Diffusion
from .unet_cond import ConditionalUNet
from .utils import seed_all


def parse_args():
    parser = argparse.ArgumentParser(description="Train Conditional Diffusion Inpainting (CD-Inpainting)")
    parser.add_argument("--images_dir", required=True, type=str)
    parser.add_argument("--labels_dir", required=True, type=str)
    parser.add_argument("--out_dir", required=True, type=str)
    parser.add_argument("--img_size", type=int, default=320)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--beta_start", type=float, default=1e-4)
    parser.add_argument("--beta_end", type=float, default=0.02)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_interval", type=int, default=1)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    seed_all(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = InfraredDataset(args.images_dir, args.labels_dir, img_size=args.img_size)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = ConditionalUNet().to(device)
    diffusion = Diffusion(
        timesteps=args.timesteps, beta_start=args.beta_start, beta_end=args.beta_end, device=device
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        total_count = 0
        for x0, cond, _mask in dataloader:
            x0 = x0.to(device)
            cond = cond.to(device)
            b = x0.shape[0]
            t = torch.randint(0, args.timesteps, (b,), device=device, dtype=torch.long)
            loss = diffusion.p_losses(model, x0, cond, t)
            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item() * b
            total_count += b

        avg_loss = total_loss / max(1, total_count)
        print(f"Epoch {epoch+1}/{args.epochs} - loss: {avg_loss:.6f}")

        if (epoch + 1) % args.save_interval == 0:
            ckpt_path = os.path.join(args.out_dir, f"ckpt_epoch_{epoch+1}.pt")
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch + 1,
                    "config": vars(args),
                },
                ckpt_path,
            )
            print(f"Saved checkpoint to {ckpt_path}")


if __name__ == "__main__":
    main()
