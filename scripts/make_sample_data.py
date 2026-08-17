"""
Generate a handful of synthetic indoor RGB + depth pairs for smoke-testing the
pipeline in environments without internet access to the real NYU Depth V2
download host (this sandbox has no route to it, only PyPI/GitHub are open).

These are NOT a substitute for NYU Depth V2 -- they exist purely so
scripts/run_eval.py's wiring (corruption -> enhancement -> depth model ->
alignment -> metrics -> benchmark -> CSV/figures) can be exercised end-to-end
before you point it at the real dataset on your own machine.

Each sample is a simple procedural "room": a back wall at a fixed distance, a
box-shaped object floor-to-ceiling closer to the camera, and a smooth depth
gradient on the floor -- enough structure to make corruption/metric code paths
non-trivial (varying depth => haze varies spatially, near-field band is
non-empty, etc.) without needing any external asset.
"""

from __future__ import annotations

import numpy as np
from PIL import Image


def make_synthetic_room(h: int = 240, w: int = 320, seed: int = 0,
                         wall_depth_m: float = 4.0, box_depth_m: float = 1.2) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)

    # --- depth map ---
    depth = np.full((h, w), wall_depth_m, dtype=np.float32)

    # floor: linear gradient from near (bottom) to wall depth (horizon line)
    horizon = int(h * 0.45)
    for y in range(horizon, h):
        t = (y - horizon) / max(h - horizon - 1, 1)
        depth[y, :] = wall_depth_m * (1 - t) + 0.25 * t  # bottom row ~0.25m (near-field band)

    # a box object, closer than the wall, roughly centred
    box_x0, box_x1 = int(w * 0.35), int(w * 0.65)
    box_y0, box_y1 = int(h * 0.35), int(h * 0.85)
    depth[box_y0:box_y1, box_x0:box_x1] = box_depth_m

    depth += rng.normal(0, 0.01, size=depth.shape).astype(np.float32)  # tiny sensor-like noise
    depth = np.clip(depth, 0.15, None)

    # --- RGB: simple shaded regions matching the depth structure, plus texture ---
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    rgb[:horizon, :] = [0.55, 0.58, 0.62]                       # wall/ceiling tone
    rgb[horizon:, :] = [0.42, 0.38, 0.33]                       # floor tone
    rgb[box_y0:box_y1, box_x0:box_x1] = [0.75, 0.35, 0.25]      # box color

    texture = rng.normal(0, 0.03, size=rgb.shape).astype(np.float32)
    rgb = np.clip(rgb + texture, 0.0, 1.0)

    return rgb, depth


def save_pair(rgb: np.ndarray, depth: np.ndarray, out_dir: str, name: str) -> None:
    import os
    os.makedirs(out_dir, exist_ok=True)
    Image.fromarray((rgb * 255).astype(np.uint8)).save(f"{out_dir}/{name}_rgb.png")
    np.save(f"{out_dir}/{name}_depth.npy", depth)


if __name__ == "__main__":
    OUT_DIR = "data/samples"
    N_SAMPLES = 4
    for i in range(N_SAMPLES):
        rgb, depth = make_synthetic_room(seed=i, wall_depth_m=3.0 + i * 0.8, box_depth_m=0.9 + i * 0.15)
        save_pair(rgb, depth, OUT_DIR, f"synthetic_{i:02d}")
        print(f"wrote {OUT_DIR}/synthetic_{i:02d}_rgb.png + _depth.npy  "
              f"(depth range {depth.min():.2f}-{depth.max():.2f} m)")
    print(f"{N_SAMPLES} synthetic indoor samples written to {OUT_DIR}/")
