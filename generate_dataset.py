
import os
import numpy as np

OUTPUT = "data/input"
os.makedirs(OUTPUT, exist_ok=True)

for i in range(200):
    y, x = np.mgrid[0:256, 0:256]
    image = np.zeros((256, 256, 3), dtype=np.uint8)

    image[:, :, 0] = (x + i * 7) % 256
    image[:, :, 1] = (y + i * 11) % 256
    image[:, :, 2] = ((x + y) // 2 + i * 13) % 256

    cx = (i * 17) % 256
    cy = (i * 23) % 256
    mask = (x - cx)**2 + (y - cy)**2 < 55**2
    image[mask] = [255, 255, 255]

    rng = np.random.default_rng(seed=i)
    noise = rng.integers(-15, 16, size=image.shape)

    image = np.clip(
        image.astype(np.int16) + noise, 0, 255
    ).astype(np.uint8)

    with open(f"{OUTPUT}/image_{i+1:03d}.ppm", "wb") as f:
        f.write(b"P6\n256 256\n255\n")
        image.tofile(f)

print("Generated 200 input images.")
