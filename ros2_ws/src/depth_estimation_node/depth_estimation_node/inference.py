"""
Self-contained model + enhancement code for the ROS 2 node -- deliberately a
standalone copy of the relevant pieces of src/depth_infer.py and src/enhance.py
from the main research repo (Robust-Monocular-Depth-Estimation-...), not an
import of it, so this package can be dropped into a robot's ros2_ws_livox/src/
without needing that repo checked out at a matching path on the Jetson.

If you change the scale-alignment math or the corruption/enhancement pipeline
in the main repo, mirror the change here -- these are copies, not symlinks.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Depth Anything V2 wrapper (see src/depth_infer.py for the full write-up of
# why the metric checkpoint still needs a scale correction)
# ---------------------------------------------------------------------------

METRIC_INDOOR_CHECKPOINTS = {
    "small": "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf",
    "base": "depth-anything/Depth-Anything-V2-Metric-Indoor-Base-hf",
    "large": "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf",
}


@dataclass
class DepthPrediction:
    raw: np.ndarray            # model output, exactly as returned (H, W), metres
    inference_seconds: float


class DepthAnythingV2Model:
    def __init__(self, size: str = "small", device: str = "cpu"):
        if size not in METRIC_INDOOR_CHECKPOINTS:
            raise ValueError(f"size must be one of {list(METRIC_INDOOR_CHECKPOINTS)}")
        self.size = size
        self.device = device
        self.model_id = METRIC_INDOOR_CHECKPOINTS[size]
        self._pipe = None

    def load(self) -> None:
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
        return DepthPrediction(raw=arr, inference_seconds=dt)


def median_scale_alignment(pred_depth: np.ndarray, ref_depth_m: np.ndarray,
                            valid_mask: np.ndarray | None = None,
                            eps: float = 1e-6) -> tuple[np.ndarray, float]:
    """Single multiplicative scale factor via median ratio against a reference
    depth map (NYU ground truth in the offline eval; the RealSense's own live
    stereo depth here). Returns (aligned_depth, scale_factor) -- the caller
    decides whether to trust and apply a freshly computed scale_factor or hold
    the last known one (see depth_node.py's stereo-confidence gating)."""
    pred_depth = np.asarray(pred_depth, dtype=np.float64)
    ref_depth_m = np.asarray(ref_depth_m, dtype=np.float64)

    if valid_mask is None:
        valid_mask = np.isfinite(ref_depth_m) & (ref_depth_m > eps)

    if not valid_mask.any():
        return pred_depth.astype(np.float32), 1.0

    scale = float(np.median(ref_depth_m[valid_mask]) / max(np.median(pred_depth[valid_mask]), eps))
    return (pred_depth * scale).astype(np.float32), scale


# ---------------------------------------------------------------------------
# Enhancement (see src/enhance.py for the full write-up, including the
# noise-amplification finding for Zero-DCE on severe low-light input)
# ---------------------------------------------------------------------------

def clahe_enhance(img: np.ndarray, clip_limit: float = 2.0, tile_grid_size: int = 8) -> np.ndarray:
    import cv2

    if img.dtype != np.uint8:
        img_u8 = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    else:
        img_u8 = img

    lab = cv2.cvtColor(img_u8, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size))
    l_eq = clahe.apply(l)

    lab_eq = cv2.merge((l_eq, a, b))
    out = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2RGB)
    return out.astype(np.float32) / 255.0


class ZeroDCENet(nn.Module):
    def __init__(self, n_iterations: int = 8):
        super().__init__()
        self.n_iterations = n_iterations
        n_feat = 32

        self.conv1 = nn.Conv2d(3, n_feat, 3, 1, 1)
        self.conv2 = nn.Conv2d(n_feat, n_feat, 3, 1, 1)
        self.conv3 = nn.Conv2d(n_feat, n_feat, 3, 1, 1)
        self.conv4 = nn.Conv2d(n_feat, n_feat, 3, 1, 1)
        self.conv5 = nn.Conv2d(n_feat * 2, n_feat, 3, 1, 1)
        self.conv6 = nn.Conv2d(n_feat * 2, n_feat, 3, 1, 1)
        self.conv7 = nn.Conv2d(n_feat * 2, 3 * n_iterations, 3, 1, 1)

        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()

    def forward(self, x: torch.Tensor):
        x1 = self.relu(self.conv1(x))
        x2 = self.relu(self.conv2(x1))
        x3 = self.relu(self.conv3(x2))
        x4 = self.relu(self.conv4(x3))
        x5 = self.relu(self.conv5(torch.cat([x3, x4], dim=1)))
        x6 = self.relu(self.conv6(torch.cat([x2, x5], dim=1)))
        curves = self.tanh(self.conv7(torch.cat([x1, x6], dim=1)))

        curve_maps = torch.split(curves, 3, dim=1)
        enhanced = x
        for a in curve_maps:
            enhanced = enhanced + a * (torch.pow(enhanced, 2) - enhanced)
        return enhanced, curves

    def load_weights(self, path: str, map_location: str = "cpu") -> None:
        state = torch.load(path, map_location=map_location)
        self.load_state_dict(state)


@torch.no_grad()
def zero_dce_enhance(img: np.ndarray, model: ZeroDCENet, device: str = "cpu") -> np.ndarray:
    model = model.to(device).eval()
    t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float().to(device)
    enhanced, _ = model(t)
    out = enhanced.squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy()
    return out.astype(np.float32)
