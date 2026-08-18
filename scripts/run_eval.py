"""
Main evaluation loop: corruption x enhancement x depth-model -> metrics CSV +
qualitative figures.

Usage on your GPU workstation or the Jetson (real run):
    python scripts/run_eval.py --data-dir data/samples --model-size small --real-model

Usage here / anywhere without HF access (wiring smoke-test, --mock-model is
the default): validates the full pipeline using a synthetic depth model that
stands in for DepthAnythingV2Model, so you can trust the wiring before the
first real run needs network access to huggingface.co.

Extending to real NYU Depth V2: replace `load_samples()` with a loader over the
official .mat file / your preferred loader; everything downstream is unchanged
since it only depends on (rgb: HWC float32 [0,1], depth_m: HW float32) pairs.
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import sys
import time

# Cap thread pools before numpy/cv2/torch spin them up -- each library defaults to
# one pool per available core, and with ~25k sequential single-image forward passes
# that oversubscription (confirmed: ~24 cores' worth of CPU time per 5s wall time on
# a run that never got past the first print statement) dominates wall-clock time far
# more than the actual GPU compute does.
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

import cv2
cv2.setNumThreads(4)

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from corruptions import apply_corruption, CORRUPTIONS          # noqa: E402
from enhance import clahe_enhance, ZeroDCENet, zero_dce_enhance  # noqa: E402
from metrics import evaluate_all                                # noqa: E402
from benchmark import benchmark_callable, write_csv, BenchmarkResult  # noqa: E402


# ---------------------------------------------------------------------------
# Mock model -- used when --real-model is not passed (e.g. in this sandbox,
# which has no route to huggingface.co). Mirrors DepthAnythingV2Model.predict's
# interface (a `.raw` disparity array + `.is_metric` flag) so run_eval.py's
# control flow is identical whether the mock or the real model is used --
# swapping --real-model in is the only change needed once you're on a machine
# with internet access.
# ---------------------------------------------------------------------------

class MockDepthModel:
    """Stands in for DepthAnythingV2Model: returns a noisy disparity map derived
    from ground truth, so the alignment/metrics/benchmark code paths get
    genuinely exercised without needing real model weights."""

    is_metric = False

    def predict(self, rgb: np.ndarray, gt_depth_m: np.ndarray, noise_sigma: float = 0.03):
        from dataclasses import dataclass
        t0 = time.perf_counter()
        true_disp = 1.0 / np.clip(gt_depth_m, 1e-3, None)
        rng = np.random.default_rng(abs(hash(rgb.tobytes())) % (2**32))
        noisy_disp = true_disp * (1 + rng.normal(0, noise_sigma, size=true_disp.shape))
        dt = time.perf_counter() - t0

        @dataclass
        class _Pred:
            raw: np.ndarray
            is_metric: bool
            inference_seconds: float

        return _Pred(raw=noisy_disp.astype(np.float32), is_metric=False, inference_seconds=dt)


def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def resolve_depth(pred, gt_depth_m: np.ndarray) -> np.ndarray:
    """Empirically (see commit history / README), the HF 'Metric-Indoor' checkpoint's
    raw output on this data source carries a consistent ~20-35% global scale bias
    against ground truth (confirmed via per-image correlation ~0.93-0.95 but
    mean-ratio consistently off) -- structure is right, absolute scale isn't. Rather
    than trust the checkpoint's metric claim at face value, apply the same
    median-scale calibration depth_infer.py already ships for exactly this case
    (see its docstring: "a sanity check on a metric checkpoint"). This is a per-image
    single multiplicative factor, so it does not erase corruption-severity error
    trends -- only removes a constant scale offset."""
    from depth_infer import least_squares_disparity_alignment, median_scale_alignment
    if pred.is_metric:
        return median_scale_alignment(pred.raw, gt_depth_m)
    return least_squares_disparity_alignment(pred.raw, gt_depth_m)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_samples(data_dir: str):
    for depth_path in sorted(glob.glob(os.path.join(data_dir, "*_depth.npy"))):
        name = os.path.basename(depth_path).replace("_depth.npy", "")
        rgb_path = os.path.join(data_dir, f"{name}_rgb.png")
        rgb = np.asarray(Image.open(rgb_path)).astype(np.float32) / 255.0
        depth = np.load(depth_path)
        yield name, rgb, depth


def load_nyu_depth_v2(split: str = "validation", limit: int | None = None):
    """Real NYU Depth V2 labeled test split (654 images), via the parquet-converted
    mirror of sayakpaul/nyu_depth_v2 on the HF Hub -- `image` is RGB uint8, `depth_map`
    is a float32 TIFF in metres (verified by direct inspection, not the .mat file, but
    same underlying Silberman et al. labeled data / same 654-image count as the
    official test split). Cached locally by `datasets` after the first call.
    """
    from datasets import load_dataset

    ds = load_dataset("sayakpaul/nyu_depth_v2", split=split, revision="refs/convert/parquet")
    n = len(ds) if limit is None else min(limit, len(ds))
    for i in range(n):
        row = ds[i]
        rgb = np.asarray(row["image"].convert("RGB")).astype(np.float32) / 255.0
        depth = np.asarray(row["depth_map"]).astype(np.float32)
        yield f"nyu_{i:04d}", rgb, depth


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

SEVERITIES = [1, 3, 5]
ENHANCEMENTS = ["none", "clahe", "zero_dce"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/samples")
    ap.add_argument("--data-source", default="synthetic", choices=["synthetic", "nyu"],
                     help="synthetic: *_rgb.png/*_depth.npy pairs under --data-dir "
                          "(wiring tests only). nyu: real NYU Depth V2 labeled test "
                          "split (654 images), downloaded/cached via the HF `datasets` "
                          "hub -- --data-dir is ignored in this mode.")
    ap.add_argument("--limit", type=int, default=None,
                     help="Cap the number of samples (e.g. for a cheap subset run "
                          "before scaling up to the full set).")
    ap.add_argument("--out-csv", default="data/results/eval_results.csv")
    ap.add_argument("--model-size", default="small", choices=["small", "base", "large"])
    ap.add_argument("--real-model", action="store_true",
                     help="Use the real Depth Anything V2 model (needs internet access "
                          "to huggingface.co -- not available in this sandbox).")
    ap.add_argument("--zero-dce-weights", default=None,
                     help="Path to trained Zero-DCE weights (see scripts/train_zero_dce.py). "
                          "If omitted, the zero_dce arm uses an untrained (random-init) net.")
    ap.add_argument("--skip-benchmark", action="store_true",
                     help="Skip the per-row timing pass (benchmark_callable). Use this for "
                          "real-model runs to avoid ~4x extra forward passes; run "
                          "src/benchmark.py separately for dedicated latency numbers.")
    args = ap.parse_args()

    if args.real_model:
        from depth_infer import DepthAnythingV2Model
        device = "cuda" if _cuda_available() else "cpu"
        model = DepthAnythingV2Model(size=args.model_size, metric=True, device=device)
        model.load()
        print(f"Loaded real model: {model.model_id} (device={device})")
    else:
        model = MockDepthModel()
        print("Using MockDepthModel (pass --real-model on a machine with internet "
              "access to run the actual Depth Anything V2 checkpoint).")

    zero_dce_net = ZeroDCENet()  # untrained by default; see scripts/train_zero_dce.py
    if args.zero_dce_weights:
        zero_dce_net.load_weights(args.zero_dce_weights)
        print(f"Loaded trained Zero-DCE weights: {args.zero_dce_weights}")

    rows = []
    benchmarks = []

    if args.data_source == "nyu":
        samples = list(load_nyu_depth_v2(limit=args.limit))
    else:
        samples = list(load_samples(args.data_dir))
        if args.limit:
            samples = samples[: args.limit]
    if not samples:
        print(f"No samples found in {args.data_dir}. Run scripts/make_sample_data.py "
              f"first, or point --data-dir at your real dataset.")
        return
    print(f"Loaded {len(samples)} samples from data-source={args.data_source}", flush=True)

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    csv_writer = None
    csv_file = open(args.out_csv, "w", newline="")
    n_rows_written = 0
    t_start = time.perf_counter()

    for sample_idx, (name, rgb, depth) in enumerate(samples):
        for corruption_name in list(CORRUPTIONS) + ["clean"]:
            for severity in (SEVERITIES if corruption_name != "clean" else [0]):
                if corruption_name == "clean":
                    corrupted = rgb
                elif corruption_name == "indoor_haze":
                    corrupted = apply_corruption(corruption_name, rgb, severity=severity, depth_m=depth)
                else:
                    corrupted = apply_corruption(corruption_name, rgb, severity=severity)

                for enh in ENHANCEMENTS:
                    if enh == "none":
                        processed = corrupted
                    elif enh == "clahe":
                        processed = clahe_enhance(corrupted)
                    else:
                        processed = zero_dce_enhance(corrupted, zero_dce_net)

                    if args.real_model:
                        from PIL import Image as PILImage
                        pil_img = PILImage.fromarray((processed * 255).astype(np.uint8))
                        pred = model.predict(pil_img)
                    else:
                        pred = model.predict(processed, depth)

                    pred_depth = resolve_depth(pred, depth)
                    metrics = evaluate_all(depth, pred_depth)

                    row = {
                        "sample": name,
                        "corruption": corruption_name,
                        "severity": severity,
                        "enhancement": enh,
                        "model_size": args.model_size,
                        "real_model": args.real_model,
                        **metrics,
                    }
                    rows.append(row)

                    if csv_writer is None:
                        csv_writer = csv.DictWriter(csv_file, fieldnames=list(row.keys()))
                        csv_writer.writeheader()
                    csv_writer.writerow(row)
                    n_rows_written += 1

                    if not args.skip_benchmark:
                        bench = benchmark_callable(
                            lambda: model.predict(processed, depth) if not args.real_model else model.predict(pil_img),
                            label=f"{corruption_name}_sev{severity}_{enh}",
                            device="cpu" if not args.real_model else "unknown",
                            n_warmup=1, n_runs=3,
                        )
                        benchmarks.append(bench)

        csv_file.flush()
        elapsed = time.perf_counter() - t_start
        rate = elapsed / (sample_idx + 1)
        remaining = rate * (len(samples) - sample_idx - 1)
        print(f"[{sample_idx + 1}/{len(samples)}] {name} done -- "
              f"{n_rows_written} rows so far, {elapsed:.0f}s elapsed, "
              f"~{remaining:.0f}s remaining", flush=True)

    csv_file.close()

    bench_csv = args.out_csv.replace(".csv", "_benchmark.csv")
    write_csv(benchmarks, bench_csv)

    print(f"\nWrote {len(rows)} result rows to {args.out_csv}")
    print(f"Wrote {len(benchmarks)} benchmark rows to {bench_csv}")

    # quick sanity summary
    abs_rels = [r["abs_rel"] for r in rows if r["corruption"] == "clean"]
    if abs_rels:
        print(f"Mean Abs Rel on CLEAN images (sanity check, should be low): {np.mean(abs_rels):.4f}")


if __name__ == "__main__":
    main()
