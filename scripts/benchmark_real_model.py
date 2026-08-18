"""
Dedicated latency/FPS benchmark for the real Depth Anything V2 model, kept
separate from run_eval.py's accuracy sweep (see run_eval.py --skip-benchmark)
so per-row timing during the big correctness sweep doesn't get 4x'd across
every corruption/severity/enhancement combo for no reason -- the model's
forward-pass latency only depends on model size and input resolution, not on
which corruption produced the pixels.

Labelled explicitly as this machine's GPU (a dev workstation), not the Jetson
Orin Nano on the Sherpa RP -- that is a separate, later benchmarking pass per
docs/GPU_JETSON_SETUP.md.
"""

from __future__ import annotations

import argparse
import os
import platform
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import torch
from PIL import Image

from benchmark import benchmark_callable, write_csv  # noqa: E402
from depth_infer import DepthAnythingV2Model          # noqa: E402
from run_eval import load_nyu_depth_v2                # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-sizes", default="small,base,large")
    ap.add_argument("--n-warmup", type=int, default=5)
    ap.add_argument("--n-runs", type=int, default=30)
    ap.add_argument("--out-csv", default="data/results/benchmark_dev_gpu.csv")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_label = f"dev-gpu:{torch.cuda.get_device_name(0)}" if device == "cuda" else "dev-cpu"
    print(f"Benchmarking on: {device_label} (host={platform.node()})")

    name, rgb, depth = next(load_nyu_depth_v2(limit=1))
    pil_img = Image.fromarray((rgb * 255).astype(np.uint8))

    results = []
    for size in args.model_sizes.split(","):
        size = size.strip()
        print(f"\nLoading {size}...")
        model = DepthAnythingV2Model(size=size, metric=True, device=device)
        model.load()

        result = benchmark_callable(
            model.predict, pil_img,
            label=f"depth-anything-v2-metric-indoor-{size}",
            device=device_label,
            n_warmup=args.n_warmup, n_runs=args.n_runs,
        )
        print(result)
        results.append(result)

        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    write_csv(results, args.out_csv)
    print(f"\nWrote {len(results)} benchmark rows to {args.out_csv}")


if __name__ == "__main__":
    main()
