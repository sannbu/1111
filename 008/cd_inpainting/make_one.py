import numpy as np
from PIL import Image, ImageDraw

H=W=320
# arkaplan: hafif noise
bg = (np.random.normal(loc=40, scale=8, size=(H,W))).clip(0,255).astype(np.uint8)
img = Image.fromarray(bg, mode="L")

# tek bbox (piksel koordinatları)
x1,y1,x2,y2 = 110,130,160,180  # width=50 height=50

# kutunun içine parlak hedef gibi bir alan çiz
draw = ImageDraw.Draw(img)
draw.rectangle([x1,y1,x2,y2], fill=200)

img.save("data/train/images/img_0001.png")
print("Saved: data/train/images/img_0001.png")

