"""Smoke-test experiment: call Slang functions from Python and save an image."""

import matplotlib.pyplot as plt
import numpy as np
import slangpy as spy

from slang_research import REPO_ROOT, create_device, load_module

# set up the gpu, opens a Metal device on M1 and compiles Slang
device = create_device()
print(f"Device: {device.info.adapter_name} ({device.info.type.name})")

module = load_module("hello.slang", device)

# Vectorized call: numpy arrays in, numpy array out.
a = np.arange(8, dtype=np.float32)
b = np.full(8, 10.0, dtype=np.float32)
print("add:", module.add(a, b, _result="numpy"))

# Per-pixel call over a 2D grid, one GPU thread per pixel
width, height = 256, 256
image = module.checkerboard(
    spy.grid(shape=(height, width)), spy.int2(width, height), 8, _result="numpy"
)
out = REPO_ROOT / "outputs" / "checkerboard.png"
plt.imsave(out, np.clip(image, 0, 1))
print("wrote", out)
