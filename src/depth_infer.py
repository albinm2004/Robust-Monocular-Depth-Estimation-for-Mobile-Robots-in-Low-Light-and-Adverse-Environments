"""
Depth Anything V2 inference wrapper, with the scale-alignment handling flagged as
the top-priority technical fix in research-plan-review.md.

Background (why this file exists as its own module instead of a one-line pipeline
call): Depth Anything V2's base "relative" checkpoints (e.g.
depth-anything/Depth-Anything-V2-Small-hf) output disparity-like, scale-and-shift
ambiguous values -- roughly proportional to inverse depth, same convention as
MiDaS -- NOT metric depth in metres. Computing Abs Rel / RMSE directly against
NYU Depth V2's metric ground truth without alignment gives meaningless numbers.
Two correct paths, both implemented below:

  1. Use a "Metric" checkpoint (e.g. depth-anything/Depth-Anything-V2-Metric-
     Indoor-Small-hf) which outputs metric depth directly -- simplest, and the
     right default for this indoor-only phase.
  2. Use a "relative" checkpoint and explicitly align it to metric depth per
     image before scoring, via least-squares scale+shift alignment in disparity
     space (the standard protocol from MiDaS / Eigen et al.).

NOTE ON THIS SANDBOX: huggingface.co is not reachable from this cloud workspace
(confirmed 403 on both the API and the resolve/CDN endpoints while building this --
GitHub and PyPI work fine, HF specifically doesn't). That means DepthAnythingV2Model
below cannot actually download weights and run *in this sandbox*. The class and the
alignment math are fully implemented and unit-testable with synthetic arrays
(see the __main__ block and scripts/run_eval.py's --self-test flag); the first real
run against actual model weights needs to happen on your GPU workstation or the
Jetson, both of which have normal internet access.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------------------
# Model wrapper
# ---------------------------------------------------------------------------

RELATIVE_CHECKPOINTS = {
    "small": "depth-anything/Depth-Anything-V2-Small-hf",
    "base": "depth-anything/Depth-Anything-V2-Base-hf",
    "large": "depth-anything/Depth-Anything-V2-Large-hf",
}

METRIC_INDOOR_CHECKPOINTS = {
    "small": "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf",
    "base": "depth-anything/Depth-Anything-V2-Metric-Indoor-Base-hf",
    "large": "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf",
}


@dataclass
class DepthPrediction:
    raw: np.ndarray            # model output, exactly as returned (H, W)
    is_metric: bool            # True if `raw` is already metres, no alignment needed
    inference_seconds: float   # for benchmark.py


class DepthAnythingV2Model:
    """
    Thin wrapper around the HF transformers depth-estimation pipeline. Kept
    intentionally minimal so it's a two-line swap between checkpoint sizes
    (accuracy-vs-latency comparison, as recommended in the plan review) and
    between relative and metric variants.
    """

    def __init__(self, size: str = "small", metric: bool = True, device: str = "cpu"):
        if size not in RELATIVE_CHECKPOINTS:
            raise ValueError(f"size must be one of {list(RELATIVE_CHECKPOINTS)}")
        self.size = size
        self.metric = metric
        self.device = device
        self.model_id = (METRIC_INDOOR_CHECKPOINTS if metric else RELATIVE_CHECKPOINTS)[size]
        self._pipe = None  # lazy-loaded

    def load(self) -> None:
        """Download + load weights. Requires internet access to huggingface.co --
        will fail in this sandbox (see module docstring); run on your GPU
        workstation or the Jetson."""
        from transformers import pipeline
        dev = 0 if self.device == "cuda" else -1
        self._pipe = pipeline(task="depth-estimation", model=self.model_id, device=dev)

    def predict(self, pil_image) -> DepthPrediction:
        if self._pipe is None:
            self.load()

        import time
        t0 = time.perf_counter()
        result = self._pipe(pil_image)
        dt = time.perf_counter() - t0

        depth_tensor = result["predicted_depth"]
        arr = depth_tensor.squeeze().detach().cpu().numpy().astype(np.float32)
        return DepthPrediction(raw=arr, is_metric=self.metric, inference_seconds=dt)


# ---------------------------------------------------------------------------
# Scale alignment (for relative / non-metric checkpoints)
# ---------------------------------------------------------------------------

def least_squares_disparity_alignment(pred_disparity: np.ndarray, gt_depth_m: np.ndarray,
                                       valid_mask: np.ndarray | None = None,
                                       eps: float = 1e-6) -> np.ndarray:
    """
    Standard MiDaS/Eigen-style alignment: relative-depth models predict something
    proportional to inverse depth (disparity). Fit an affine map in disparity
    space -- disparity_hat = a * pred + b -- by least squares against 1/gt_depth,
    then invert to get an aligned metric depth map.

    pred_disparity: (H, W) raw model output (relative, unitless)
    gt_depth_m:     (H, W) ground-truth depth in metres (0 or NaN where invalid)
    valid_mask:     optional (H, W) bool; defaults to gt_depth_m > 0 and finite

    Returns an (H, W) depth map in metres, same domain as gt_depth_m.
    """
    pred_disparity = np.asarray(pred_disparity, dtype=np.float64)
    gt_depth_m = np.asarray(gt_depth_m, dtype=np.float64)

    if valid_mask is None:
        valid_mask = np.isfinite(gt_depth_m) & (gt_depth_m > eps)

    gt_disp = 1.0 / np.clip(gt_depth_m[valid_mask], eps, None)
    pred_vals = pred_disparity[valid_mask]

    # solve [a, b] minimising ||a*pred + b - gt_disp||^2
    A = np.stack([pred_vals, np.ones_like(pred_vals)], axis=1)
    solution, *_ = np.linalg.lstsq(A, gt_disp, rcond=None)
    a, b = solution

    aligned_disp = a * pred_disparity + b
    aligned_disp = np.clip(aligned_disp, eps, None)  # guard against non-positive disparity
    aligned_depth = 1.0 / aligned_disp
    return aligned_depth.astype(np.float32)


def median_scale_alignment(pred_depth: np.ndarray, gt_depth_m: np.ndarray,
                            valid_mask: np.ndarray | None = None,
                            eps: float = 1e-6) -> np.ndarray:
    """
    Simpler alternative: single multiplicative scale factor via median ratio.
    Appropriate when `pred_depth` is already depth-proportional (not
    disparity-proportional) but off by an unknown constant scale -- e.g. as a
    sanity check on a metric checkpoint, or a lighter-weight ablation to report
    alongside the full least-squares disparity alignment.
    """
    pred_depth = np.asarray(pred_depth, dtype=np.float64)
    gt_depth_m = np.asarray(gt_depth_m, dtype=np.float64)

    if valid_mask is None:
        valid_mask = np.isfinite(gt_depth_m) & (gt_depth_m > eps)

    scale = np.median(gt_depth_m[valid_mask]) / max(np.median(pred_depth[valid_mask]), eps)
    return (pred_depth * scale).astype(np.float32)


def resolve_to_metric_depth(prediction: DepthPrediction, gt_depth_m: np.ndarray,
                             valid_mask: np.ndarray | None = None) -> np.ndarray:
    """Single entry point scripts/run_eval.py should call: handles both the
    metric-checkpoint (no-op) and relative-checkpoint (align) cases so the rest
    of the pipeline never has to think about which checkpoint is loaded."""
    if prediction.is_metric:
        return prediction.raw.astype(np.float32)
    return least_squares_disparity_alignment(prediction.raw, gt_depth_m, valid_mask)


if __name__ == "__main__":
    # Self-test of the alignment math only -- no network / model weights needed.
    # Construct a synthetic ground-truth depth map and a synthetic "relative"
    # prediction that is a known affine function of 1/depth, then verify the
    # alignment recovers the true depth to within numerical tolerance.
    rng = np.random.default_rng(0)
    gt = rng.uniform(0.3, 6.0, size=(64, 64)).astype(np.float32)  # metres, indoor range
    true_disp = 1.0 / gt
    a_true, b_true = 2.7, 0.15
    synthetic_pred_disp = a_true * true_disp + b_true
    synthetic_pred_disp += rng.normal(0, 0.001, size=synthetic_pred_disp.shape)  # tiny noise

    recovered = least_squares_disparity_alignment(synthetic_pred_disp, gt)
    err = np.abs(recovered - gt).mean()
    print(f"mean abs error after alignment: {err:.5f} m (should be small, e.g. < 0.05)")
    assert err < 0.05, "alignment self-test failed"
    print("depth_infer.py alignment self-test: OK")
