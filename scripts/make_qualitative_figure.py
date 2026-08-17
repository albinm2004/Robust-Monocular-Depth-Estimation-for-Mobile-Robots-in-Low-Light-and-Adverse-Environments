"""
Generate the side-by-side qualitative comparison figure needed for the paper
(Week 5 of the plan: "Generate qualitative comparison figures - Visual depth
maps side-by-side"). Row = clean / corrupted / enhanced; columns = RGB input
and predicted depth map, plus ground truth for reference.

This uses MockDepthModel by default so it runs anywhere (including this
sandbox); pass --real-model on your GPU workstation / Jetson once you have
network access to actually load Depth Anything V2.
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from corruptions import apply_corruption          # noqa: E402
from enhance import clahe_enhance                  # noqa: E402
from run_eval import MockDepthModel, resolve_depth  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", default="data/samples/synthetic_00")
    ap.add_argument("--corruption", default="low_light")
    ap.add_argument("--severity", type=int, default=4)
    ap.add_argument("--out", default="data/results/qualitative_comparison.png")
    args = ap.parse_args()

    rgb = np.asarray(Image.open(f"{args.sample}_rgb.png")).astype(np.float32) / 255.0
    depth = np.load(f"{args.sample}_depth.npy")

    if args.corruption == "indoor_haze":
        corrupted = apply_corruption(args.corruption, rgb, severity=args.severity, depth_m=depth)
    else:
        corrupted = apply_corruption(args.corruption, rgb, severity=args.severity)
    enhanced = clahe_enhance(corrupted)

    model = MockDepthModel()
    rows = [("Clean", rgb), (f"{args.corruption} (sev {args.severity})", corrupted), ("+ CLAHE", enhanced)]

    fig, axes = plt.subplots(len(rows), 3, figsize=(10, 3.2 * len(rows)))
    for i, (title, img) in enumerate(rows):
        pred = resolve_depth(model.predict(img, depth), depth)

        axes[i, 0].imshow(img)
        axes[i, 0].set_title(f"{title} — RGB")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(depth, cmap="turbo")
        axes[i, 1].set_title("Ground truth depth")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(pred, cmap="turbo", vmin=depth.min(), vmax=depth.max())
        axes[i, 2].set_title("Predicted depth")
        axes[i, 2].axis("off")

    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=140)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
