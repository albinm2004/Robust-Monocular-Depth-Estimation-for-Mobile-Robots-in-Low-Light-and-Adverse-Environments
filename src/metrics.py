"""
Depth evaluation metrics: standard Abs Rel / RMSE / delta1, plus the near-field
band metric recommended in research-plan-review.md (ties results directly to the
Sherpa RP RealSense D435i's documented obstacle-detection range of 0.25-0.70 m).
"""

from __future__ import annotations

import numpy as np


def _valid(gt: np.ndarray, pred: np.ndarray, mask: np.ndarray | None, eps: float) -> np.ndarray:
    m = np.isfinite(gt) & np.isfinite(pred) & (gt > eps) & (pred > eps)
    if mask is not None:
        m &= mask
    return m


def abs_rel(gt: np.ndarray, pred: np.ndarray, mask: np.ndarray | None = None, eps: float = 1e-6) -> float:
    m = _valid(gt, pred, mask, eps)
    if not m.any():
        return float("nan")
    return float(np.mean(np.abs(gt[m] - pred[m]) / gt[m]))


def rmse(gt: np.ndarray, pred: np.ndarray, mask: np.ndarray | None = None, eps: float = 1e-6) -> float:
    m = _valid(gt, pred, mask, eps)
    if not m.any():
        return float("nan")
    return float(np.sqrt(np.mean((gt[m] - pred[m]) ** 2)))


def delta1(gt: np.ndarray, pred: np.ndarray, mask: np.ndarray | None = None, eps: float = 1e-6) -> float:
    """Fraction of pixels where max(gt/pred, pred/gt) < 1.25 -- standard threshold accuracy."""
    m = _valid(gt, pred, mask, eps)
    if not m.any():
        return float("nan")
    ratio = np.maximum(gt[m] / pred[m], pred[m] / gt[m])
    return float(np.mean(ratio < 1.25))


def near_field_band_mask(gt_depth_m: np.ndarray, low_m: float = 0.25, high_m: float = 0.70) -> np.ndarray:
    """
    Boolean mask selecting the RealSense D435i's configured obstacle-detection
    band (see Sherpa RP manual p.16). Intersect this with your usual valid-depth
    mask before calling abs_rel/rmse/delta1 to get a near-field-only score --
    the number that best reflects depth quality where it actually matters for
    the robot's blind-spot obstacle avoidance.
    """
    return (gt_depth_m >= low_m) & (gt_depth_m <= high_m)


def evaluate_all(gt_depth_m: np.ndarray, pred_depth_m: np.ndarray,
                  valid_mask: np.ndarray | None = None) -> dict:
    """Convenience wrapper returning the full-frame metrics plus the near-field
    band metrics side by side, in the shape scripts/run_eval.py logs to CSV."""
    near_mask = near_field_band_mask(gt_depth_m)
    if valid_mask is not None:
        near_mask = near_mask & valid_mask

    return {
        "abs_rel": abs_rel(gt_depth_m, pred_depth_m, valid_mask),
        "rmse": rmse(gt_depth_m, pred_depth_m, valid_mask),
        "delta1": delta1(gt_depth_m, pred_depth_m, valid_mask),
        "abs_rel_near_field_0.25_0.70m": abs_rel(gt_depth_m, pred_depth_m, near_mask),
        "rmse_near_field_0.25_0.70m": rmse(gt_depth_m, pred_depth_m, near_mask),
        "delta1_near_field_0.25_0.70m": delta1(gt_depth_m, pred_depth_m, near_mask),
        "near_field_pixel_count": int(near_mask.sum()),
    }


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    gt = rng.uniform(0.1, 5.0, size=(64, 64)).astype(np.float32)
    pred = gt + rng.normal(0, 0.05, size=gt.shape).astype(np.float32)  # small, realistic error
    pred = np.clip(pred, 1e-3, None)

    results = evaluate_all(gt, pred)
    for k, v in results.items():
        print(f"{k}: {v}")

    assert results["abs_rel"] < 0.1, "sanity check failed: abs_rel too high for near-perfect prediction"
    assert results["delta1"] > 0.9, "sanity check failed: delta1 too low for near-perfect prediction"
    print("metrics.py self-test: OK")
