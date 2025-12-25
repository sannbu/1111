# Paper Alignment Notes

This file documents minimal fixes made where the provided table/text was ambiguous while keeping the UNet, condition concatenation, and DDPM logic faithful.

1) **Padding choice**: The table occasionally mentions “P2”; all convolutions use `padding=1` for 3×3 kernels to preserve spatial sizes at every level. This keeps skip shapes consistent.
2) **LocalAttention details**: The paper did not specify window size or heads. Implemented a 2D local window attention with `window_size=8`, `num_heads=4` (all channel dimensions in the network are divisible by 4). Assertions ensure spatial dims are divisible by the window size.
3) **Timestep embedding**: The table omits timestep conditioning but DDPM requires it. A sinusoidal embedding followed by MLP is added and injected (as a bias) into feature maps at each stage. This preserves the listed operator order while enabling time conditioning.
4) **Decoder channel ambiguities**:
   - **D3**: The table lists `in=(512 + 256)+1` before a Conv, then `Cat(skip, current)` and a Conv expecting 512 input. To avoid double-using the skip and to satisfy the second Conv’s 512 input, the implementation uses `Conv(upsampled 512 + cond 1 → 256)`, then concatenates the E4 skip (256) before the second Conv (256+256→128). Assertions guard the channel counts.
   - **D2**: Interpreted literally: Upsampled D3 (128) is concatenated with E3 skip (256) and condition (1) before Conv1 (→256); no extra Cat after Conv2.
   - **D1**: Similar ambiguity to D3. Implemented `Conv(upsampled 256 + cond 1 → 128)`, then concatenated the E2 skip (128) to reach the expected 256 channels before the final Conv (256→64). Assertions enforce this path.
5) **Unused E1 skip**: The table does not specify a skip connection for the first encoder block. E1 features are kept but not concatenated in the decoder, matching the provided decoder channel arithmetic.

All concatenations have runtime assertions to surface any mismatch between expected and actual channel counts.
