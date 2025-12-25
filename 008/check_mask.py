import numpy as np
from PIL import Image

bg_path   = "cd_inpainting/data/train/images/img_0001.png"
out_path  = "outputs/out_mask0.png"        # test edeceğin çıktı
mask_path = "outputs/debug_mask.png"

bg   = np.array(Image.open(bg_path).convert("L"), dtype=np.float32)
out  = np.array(Image.open(out_path).convert("L"), dtype=np.float32)
mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32) / 255.0

M = (mask > 0.5).astype(np.float32)
diff = np.abs(out - bg)

outside = diff[M < 0.5]
inside  = diff[M > 0.5]

def stats(name, arr):
    if arr.size == 0:
        print(f"{name}: EMPTY (no pixels)")
        return
    print(f"{name}:")
    print("  mean abs diff:", float(arr.mean()))
    print("  max  abs diff:", float(arr.max()))

stats("Outside mask", outside)
stats("Inside mask", inside)

# fark görseli
diff_img = (diff / max(1.0, diff.max()) * 255).astype(np.uint8)
Image.fromarray(diff_img).save("outputs/diff.png")
print("Saved outputs/diff.png")
