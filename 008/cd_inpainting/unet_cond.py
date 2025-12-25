from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_timestep_embedding(timesteps: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    """Sinusoidal timestep embeddings."""
    half = dim // 2
    freqs = torch.exp(-torch.arange(0, half, dtype=torch.float32, device=timesteps.device) * (torch.log(torch.tensor(max_period)) / half))
    args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=1)
    return embedding


class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int):
        super().__init__()
        self.ln = nn.LayerNorm(num_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        orig_shape = x.shape
        x = x.permute(0, 2, 3, 1)
        x = self.ln(x)
        x = x.permute(0, 3, 1, 2)
        assert x.shape == orig_shape, f"LayerNorm2d shape mismatch: {orig_shape} -> {x.shape}"
        return x


class LocalAttention(nn.Module):
    """Local window self-attention."""

    def __init__(self, dim: int, window_size: int = 8, num_heads: int = 4):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.norm = LayerNorm2d(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        assert c == self.dim, f"LocalAttention expected {self.dim} channels, got {c}"
        ws = self.window_size
        assert h % ws == 0 and w % ws == 0, f"LocalAttention window {ws} must divide spatial dims ({h},{w})"
        x_norm = self.norm(x)
        # reshape into windows
        x_reshaped = x_norm.view(b, c, h // ws, ws, w // ws, ws)
        x_reshaped = x_reshaped.permute(0, 2, 4, 3, 5, 1)  # (b, num_h, num_w, ws, ws, c)
        windows = x_reshaped.reshape(-1, ws * ws, c)  # (b*num_windows, tokens, c)
        attended, _ = self.attn(windows, windows, windows)
        attended = attended.view(b, h // ws, w // ws, ws, ws, c)
        attended = attended.permute(0, 5, 1, 3, 2, 4).reshape(b, c, h, w)
        return x + attended  # residual


class ConditionalUNet(nn.Module):
    """Image-conditioned UNet with multi-scale condition concatenation."""

    def __init__(self, img_channels: int = 1, cond_channels: int = 1, time_dim: int = 256, window_size: int = 8):
        super().__init__()
        self.img_channels = img_channels
        self.cond_channels = cond_channels
        self.window_size = window_size

        # time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, 512),
            nn.GELU(),
            nn.Linear(512, 512),
        )
        self.to_time64 = nn.Linear(512, 64)
        self.to_time128 = nn.Linear(512, 128)
        self.to_time256 = nn.Linear(512, 256)
        self.to_time512 = nn.Linear(512, 512)

        # Encoder
        self.e1_conv = nn.Conv2d(img_channels + cond_channels, 64, kernel_size=3, padding=1)

        self.e2_conv1 = nn.Conv2d(64 + cond_channels, 128, kernel_size=3, padding=1)
        self.e2_ln1 = LayerNorm2d(128)
        self.e2_conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.e2_ln2 = LayerNorm2d(128)
        self.pool2 = nn.MaxPool2d(2)

        self.e3_conv1 = nn.Conv2d(128 + cond_channels, 256, kernel_size=3, padding=1)
        self.e3_ln1 = LayerNorm2d(256)
        self.e3_conv2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.e3_ln2 = LayerNorm2d(256)
        self.pool3 = nn.MaxPool2d(2)

        self.e4_conv1 = nn.Conv2d(256 + cond_channels, 256, kernel_size=3, padding=1)
        self.e4_ln1 = LayerNorm2d(256)
        self.e4_conv2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.e4_ln2 = LayerNorm2d(256)
        self.pool4 = nn.MaxPool2d(2)

        # Bridge
        self.b_conv1 = nn.Conv2d(256 + cond_channels, 512, kernel_size=3, padding=1)
        self.b_ln1 = LayerNorm2d(512)
        self.b_conv2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.b_ln2 = LayerNorm2d(512)
        self.b_attn = LocalAttention(512, window_size=window_size)

        # Decoder
        self.up3 = nn.Upsample(scale_factor=2, mode="nearest")
        self.d3_conv1 = nn.Conv2d(512 + cond_channels, 256, kernel_size=3, padding=1)
        self.d3_conv2 = nn.Conv2d(256 + 256, 128, kernel_size=3, padding=1)
        self.d3_attn = LocalAttention(128, window_size=window_size)

        self.up2 = nn.Upsample(scale_factor=2, mode="nearest")
        self.d2_conv1 = nn.Conv2d(128 + 256 + cond_channels, 256, kernel_size=3, padding=1)
        self.d2_ln1 = LayerNorm2d(256)
        self.d2_conv2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.d2_attn = LocalAttention(256, window_size=window_size)

        self.up1 = nn.Upsample(scale_factor=2, mode="nearest")
        self.d1_conv1 = nn.Conv2d(256 + cond_channels, 128, kernel_size=3, padding=1)
        self.d1_conv2 = nn.Conv2d(128 + 128, 64, kernel_size=3, padding=1)
        self.d1_attn = LocalAttention(64, window_size=window_size)

        self.out_conv = nn.Conv2d(64, 1, kernel_size=3, padding=1)

    def _inject_time(self, x: torch.Tensor, t_embed: torch.Tensor, proj: nn.Linear) -> torch.Tensor:
        time_feat = proj(t_embed).unsqueeze(-1).unsqueeze(-1)
        return x + time_feat

    def _resize_cond(self, cond: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.interpolate(cond, size=target.shape[-2:], mode="nearest")

    def forward(self, x: torch.Tensor, cond: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: noisy image (B,1,H,W)
            cond: condition image (B,1,H,W)
            t: timestep tensor (B,) int or float
        """
        if t.dim() == 0:
            t = t.unsqueeze(0)
        t_embed = get_timestep_embedding(t, 256)
        t_embed = self.time_mlp(t_embed)

        # Encoder E1
        cond_e1 = self._resize_cond(cond, x)
        e1_in = torch.cat([x, cond_e1], dim=1)
        assert e1_in.shape[1] == self.img_channels + self.cond_channels, f"E1 concat channels {e1_in.shape}"
        e1 = self.e1_conv(e1_in)
        e1 = self._inject_time(e1, t_embed, self.to_time64)
        skip1 = e1  # not used in decoder but kept for completeness

        # Encoder E2
        cond_e2 = self._resize_cond(cond, e1)
        e2_in = torch.cat([e1, cond_e2], dim=1)
        assert e2_in.shape[1] == 64 + self.cond_channels, f"E2 concat channels {e2_in.shape}"
        e2 = self.e2_conv1(e2_in)
        e2 = self.e2_ln1(e2)
        e2 = F.gelu(e2)
        e2 = self._inject_time(e2, t_embed, self.to_time128)
        e2 = self.e2_conv2(e2)
        e2 = self.e2_ln2(e2)
        skip2 = e2
        e2 = self.pool2(e2)

        # Encoder E3
        cond_e3 = self._resize_cond(cond, e2)
        e3_in = torch.cat([e2, cond_e3], dim=1)
        assert e3_in.shape[1] == 128 + self.cond_channels, f"E3 concat channels {e3_in.shape}"
        e3 = self.e3_conv1(e3_in)
        e3 = self.e3_ln1(e3)
        e3 = F.gelu(e3)
        e3 = self._inject_time(e3, t_embed, self.to_time256)
        e3 = self.e3_conv2(e3)
        e3 = self.e3_ln2(e3)
        skip3 = e3
        e3 = self.pool3(e3)

        # Encoder E4
        cond_e4 = self._resize_cond(cond, e3)
        e4_in = torch.cat([e3, cond_e4], dim=1)
        assert e4_in.shape[1] == 256 + self.cond_channels, f"E4 concat channels {e4_in.shape}"
        e4 = self.e4_conv1(e4_in)
        e4 = self.e4_ln1(e4)
        e4 = F.gelu(e4)
        e4 = self._inject_time(e4, t_embed, self.to_time256)
        e4 = self.e4_conv2(e4)
        e4 = self.e4_ln2(e4)
        skip4 = e4
        e4 = self.pool4(e4)

        # Bridge
        cond_b = self._resize_cond(cond, e4)
        b_in = torch.cat([e4, cond_b], dim=1)
        assert b_in.shape[1] == 256 + self.cond_channels, f"B concat channels {b_in.shape}"
        b = self.b_conv1(b_in)
        b = self.b_ln1(b)
        b = F.gelu(b)
        b = self._inject_time(b, t_embed, self.to_time512)
        b = self.b_conv2(b)
        b = self.b_ln2(b)
        b = F.gelu(b)
        b = self.b_attn(b)

        # Decoder D3
        d3 = self.up3(b)
        cond_d3 = self._resize_cond(cond, d3)
        d3_in = torch.cat([d3, cond_d3], dim=1)
        expected_d3_in = 512 + self.cond_channels
        assert d3_in.shape[1] == expected_d3_in, f"D3 pre-skip concat channels {d3_in.shape}"
        d3 = self.d3_conv1(d3_in)
        d3 = self._inject_time(d3, t_embed, self.to_time256)
        # Cat skip from encoder level (skip4)
        assert skip4.shape[2:] == d3.shape[2:], f"D3 spatial mismatch skip4 {skip4.shape} vs d3 {d3.shape}"
        d3 = torch.cat([skip4, d3], dim=1)
        assert d3.shape[1] == 256 + 256, f"D3 post-skip channels {d3.shape}"
        d3 = self.d3_conv2(d3)
        d3 = self.d3_attn(d3)

        # Decoder D2
        d2 = self.up2(d3)
        cond_d2 = self._resize_cond(cond, d2)
        # here include skip3 before conv as specified
        assert skip3.shape[2:] == d2.shape[2:], f"D2 spatial mismatch skip3 {skip3.shape} vs d2 {d2.shape}"
        d2_in = torch.cat([d2, skip3, cond_d2], dim=1)
        expected_d2_in = 128 + 256 + self.cond_channels
        assert d2_in.shape[1] == expected_d2_in, f"D2 concat channels {d2_in.shape}"
        d2 = self.d2_conv1(d2_in)
        d2 = self.d2_ln1(d2)
        d2 = F.gelu(d2)
        d2 = self._inject_time(d2, t_embed, self.to_time256)
        d2 = self.d2_conv2(d2)
        d2 = self.d2_attn(d2)

        # Decoder D1
        d1 = self.up1(d2)
        cond_d1 = self._resize_cond(cond, d1)
        d1_in = torch.cat([d1, cond_d1], dim=1)
        expected_d1_in = 256 + self.cond_channels
        assert d1_in.shape[1] == expected_d1_in, f"D1 pre-skip concat channels {d1_in.shape}"
        d1 = self.d1_conv1(d1_in)
        d1 = self._inject_time(d1, t_embed, self.to_time128)
        # Cat skip from encoder level (skip2)
        assert skip2.shape[2:] == d1.shape[2:], f"D1 spatial mismatch skip2 {skip2.shape} vs d1 {d1.shape}"
        d1 = torch.cat([skip2, d1], dim=1)
        assert d1.shape[1] == 128 + 128, f"D1 post-skip channels {d1.shape}"
        d1 = self.d1_conv2(d1)
        d1 = self.d1_attn(d1)

        out = self.out_conv(d1)
        return out
