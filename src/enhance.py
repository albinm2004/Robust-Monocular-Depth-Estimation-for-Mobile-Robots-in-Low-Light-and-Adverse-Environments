"""
Low-light enhancement preprocessing: CLAHE and Zero-DCE.

CLAHE (clahe_enhance) works immediately, no training required -- it's a classical
contrast-limited histogram equalization applied in LAB space so color isn't distorted.

Zero-DCE (ZeroDCENet + zero_dce_enhance) is a learned curve-estimation network.
It is trained WITHOUT paired/reference images (the original Zero-DCE paper's whole
point), using four non-reference losses computed directly on the enhanced output:
spatial consistency, exposure control, color constancy, and illumination smoothness.
That means you can train it directly on a folder of your own low-light Sherpa RP
captures with no ground-truth "clean" counterpart needed -- see
scripts/train_zero_dce.py.

Note: this sandbox has no route to huggingface.co or github release assets for
pretrained Zero-DCE weights (network allowlist blocks it -- confirmed while building
this). The network below is a fresh (randomly initialised) DCE-Net until you either
(a) train it briefly with scripts/train_zero_dce.py on your own machine, or
(b) point ZeroDCENet.load_weights() at the official pretrained .pth if you fetch it
yourself from https://github.com/Li-Chongyi/Zero-DCE.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# CLAHE -- classical, no training needed
# ---------------------------------------------------------------------------

def clahe_enhance(img: np.ndarray, clip_limit: float = 2.0, tile_grid_size: int = 8) -> np.ndarray:
    """
    Apply CLAHE to the L channel in LAB space (preserves color better than
    applying CLAHE per-RGB-channel independently).

    img: float32 [0,1] or uint8 [0,255] RGB array, (H, W, 3).
    Returns float32 [0,1] RGB array.
    """
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


# ---------------------------------------------------------------------------
# Zero-DCE -- lightweight curve-estimation net (DCE-Net), unsupervised training
# ---------------------------------------------------------------------------

class ZeroDCENet(nn.Module):
    """
    7-layer conv net predicting per-pixel, per-channel curve parameters
    (n_iterations x 3 channels) as in the Zero-DCE paper. Iteratively applies
    the light-enhancement curve LE(x) = x + A*x*(1-x) to the input.
    """

    def __init__(self, n_iterations: int = 8):
        super().__init__()
        self.n_iterations = n_iterations
        n_feat = 32

        self.conv1 = nn.Conv2d(3, n_feat, 3, 1, 1)
        self.conv2 = nn.Conv2d(n_feat, n_feat, 3, 1, 1)
        self.conv3 = nn.Conv2d(n_feat, n_feat, 3, 1, 1)
        self.conv4 = nn.Conv2d(n_feat, n_feat, 3, 1, 1)
        self.conv5 = nn.Conv2d(n_feat * 2, n_feat, 3, 1, 1)  # skip connection
        self.conv6 = nn.Conv2d(n_feat * 2, n_feat, 3, 1, 1)  # skip connection
        self.conv7 = nn.Conv2d(n_feat * 2, 3 * n_iterations, 3, 1, 1)

        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x1 = self.relu(self.conv1(x))
        x2 = self.relu(self.conv2(x1))
        x3 = self.relu(self.conv3(x2))
        x4 = self.relu(self.conv4(x3))
        x5 = self.relu(self.conv5(torch.cat([x3, x4], dim=1)))
        x6 = self.relu(self.conv6(torch.cat([x2, x5], dim=1)))
        curves = self.tanh(self.conv7(torch.cat([x1, x6], dim=1)))

        curve_maps = torch.split(curves, 3, dim=1)  # n_iterations tensors of (B,3,H,W)
        enhanced = x
        for a in curve_maps:
            enhanced = enhanced + a * (torch.pow(enhanced, 2) - enhanced)
        return enhanced, curves

    def load_weights(self, path: str, map_location: str = "cpu") -> None:
        state = torch.load(path, map_location=map_location)
        self.load_state_dict(state)


@torch.no_grad()
def zero_dce_enhance(img: np.ndarray, model: ZeroDCENet, device: str = "cpu") -> np.ndarray:
    """Run a (trained or untrained) ZeroDCENet on a single HWC float32 [0,1] image."""
    model = model.to(device).eval()
    t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float().to(device)
    enhanced, _ = model(t)
    out = enhanced.squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy()
    return out.astype(np.float32)


# ---------------------------------------------------------------------------
# Non-reference losses (needed by scripts/train_zero_dce.py; exposed here so
# they can be unit tested / reused independently of the training loop)
# ---------------------------------------------------------------------------

def color_constancy_loss(enhanced: torch.Tensor) -> torch.Tensor:
    mean_rgb = enhanced.mean(dim=(2, 3))  # (B, 3)
    mr, mg, mb = mean_rgb[:, 0], mean_rgb[:, 1], mean_rgb[:, 2]
    return ((mr - mg) ** 2 + (mr - mb) ** 2 + (mb - mg) ** 2).mean()


def exposure_loss(enhanced: torch.Tensor, patch_size: int = 16, e_val: float = 0.6) -> torch.Tensor:
    gray = enhanced.mean(dim=1, keepdim=True)
    pooled = F.avg_pool2d(gray, patch_size)
    return ((pooled - e_val) ** 2).mean()


def illumination_smoothness_loss(curves: torch.Tensor) -> torch.Tensor:
    dh = curves[:, :, 1:, :] - curves[:, :, :-1, :]
    dw = curves[:, :, :, 1:] - curves[:, :, :, :-1]
    return (dh.pow(2).mean() + dw.pow(2).mean())


def spatial_consistency_loss(enhanced: torch.Tensor, original: torch.Tensor) -> torch.Tensor:
    """Simplified spatial-consistency term: penalise large local-gradient
    differences between input and enhanced output so structure is preserved."""
    def grad(x):
        gx = x[:, :, :, 1:] - x[:, :, :, :-1]
        gy = x[:, :, 1:, :] - x[:, :, :-1, :]
        return gx, gy

    e_gray = enhanced.mean(dim=1, keepdim=True)
    o_gray = original.mean(dim=1, keepdim=True)
    egx, egy = grad(e_gray)
    ogx, ogy = grad(o_gray)
    return (F.l1_loss(egx, ogx) + F.l1_loss(egy, ogy))
