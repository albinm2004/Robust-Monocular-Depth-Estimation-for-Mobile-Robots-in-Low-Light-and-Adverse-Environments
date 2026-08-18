"""
Generate the side-by-side qualitative comparison figure needed for the paper
(Week 5 of the plan: "Generate qualitative comparison figures - Visual depth
maps side-by-side"). Row = clean / corrupted / enhanced; columns = RGB input
and predicted depth map, plus ground truth for reference.

Uses MockDepthModel by default so it runs anywhere without network/GPU access;
pass --real-model (needs the same huggingface.co access as run_eval.py) to use
the actual Depth Anything V2 checkpoint instead.
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

from corruptions import apply_corruption                       # noqa: E402
from enhance import clahe_enhance, ZeroDCENet, zero_dce_enhance  # noqa: E402
from run_eval import MockDepthModel, resolve_depth, load_nyu_depth_v2  # noqa: E402


def _load_sample(args):
    if args.data_source == "nyu":
        gen = load_nyu_depth_v2(limit=args.nyu_index + 1)
        *_, (name, rgb, depth) = gen
        return rgb, depth
    rgb = np.asarray(Image.open(f"{args.sample}_rgb.png")).astype(np.float32) / 255.0
    depth = np.load(f"{args.sample}_depth.npy")
    return rgb, depth


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-source", default="synthetic", choices=["synthetic", "nyu"])
    ap.add_argument("--sample", default="data/samples/synthetic_00",
                     help="Used when --data-source synthetic.")
    ap.add_argument("--nyu-index", type=int, default=0,
                     help="Index into the NYU validation split, used when --data-source nyu.")
    ap.add_argument("--corruption", default="low_light")
    ap.add_argument("--severity", type=int, default=4)
    ap.add_argument("--model-size", default="small", choices=["small", "base", "large"])
    ap.add_argument("--real-model", action="store_true")
    ap.add_argument("--zero-dce-weights", default=None)
    ap.add_argument("--out", default="data/results/qualitative_comparison.png")
    args = ap.parse_args()

    rgb, depth = _load_sample(args)

    if args.corruption == "indoor_haze":
        corrupted = apply_corruption(args.corruption, rgb, severity=args.severity, depth_m=depth)
    else:
        corrupted = apply_corruption(args.corruption, rgb, severity=args.severity)

    zero_dce_net = ZeroDCENet()
    if args.zero_dce_weights:
        zero_dce_net.load_weights(args.zero_dce_weights)
    clahe = clahe_enhance(corrupted)
    zero_dce = zero_dce_enhance(corrupted, zero_dce_net)

    if args.real_model:
        from depth_infer import DepthAnythingV2Model
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = DepthAnythingV2Model(size=args.model_size, metric=True, device=device)
        model.load()

        def predict(img):
            pil_img = Image.fromarray((img * 255).astype(np.uint8))
            return resolve_depth(model.predict(pil_img), depth)
    else:
        model = MockDepthModel()

        def predict(img):
            return resolve_depth(model.predict(img, depth), depth)

    rows = [
        ("Clean", rgb),
        (f"{args.corruption} (sev {args.severity})", corrupted),
        ("+ CLAHE", clahe),
        ("+ Zero-DCE", zero_dce),
    ]

    fig, axes = plt.subplots(len(rows), 3, figsize=(10, 3.2 * len(rows)))
    for i, (title, img) in enumerate(rows):
        pred = predict(img)

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
