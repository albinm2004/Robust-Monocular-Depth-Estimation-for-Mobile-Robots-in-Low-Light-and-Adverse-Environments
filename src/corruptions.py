"""
Indoor-scoped corruption pipeline for adverse-condition monocular depth estimation.

Scope note: this project phase is indoor-only (see research-plan-review.md).
Corruptions here are chosen to be physically plausible for an indoor mobile-robot
setting (dim/uneven lighting, indoor haze/smoke/dust, sensor noise/blur). Outdoor-
specific effects (rain streaks, atmospheric fog at outdoor scale, sun glare) are
intentionally NOT implemented here -- they belong to the later outdoor phase.

All functions take an RGB image as a float32 numpy array in [0, 1], shape (H, W, 3),
and return a corrupted image in the same format. The haze/smoke corruption additionally
takes a depth map (H, W) in metres, since indoor haze/smoke synthesis uses the same
depth-based scattering model as outdoor fog synthesis -- only the physical
interpretation of the label changes, not the math.

Severity is expressed on a 1-5 scale across all corruption types so that results can
be reported against a single consistent x-axis (see metrics.py / scripts/run_eval.py).
"""

from __future__ import annotations

import numpy as np


def _to_float01(img: np.ndarray) -> np.ndarray:
    """Accept uint8 [0,255] or float [0,1] input; always return float32 [0,1]."""
    img = np.asarray(img)
    if img.dtype == np.uint8:
        return img.astype(np.float32) / 255.0
    img = img.astype(np.float32)
    if img.max() > 1.5:
        img = img / 255.0
    return np.clip(img, 0.0, 1.0)


def low_light(img: np.ndarray, severity: int = 3, add_noise: bool = True,
              rng: np.random.Generator | None = None) -> np.ndarray:
    """
    Simulate dim/uneven indoor lighting (e.g. corridor lights off, dusk through
    windows, single desk lamp).

    Mechanism: gamma darkening (mimics under-exposure) + optional low-light sensor
    noise (real cameras get noisier as gain increases in the dark), matching common
    practice in low-light vision benchmarks (e.g. the gamma+noise recipe used to
    build LOL-style synthetic pairs).

    severity 1 (barely dim) .. 5 (near-dark, only exit-sign-level light).
    """
    rng = rng or np.random.default_rng()
    img = _to_float01(img)

    # gamma > 1 darkens midtones/shadows faster than highlights, which is what
    # under-exposure looks like in practice.
    gamma = 1.0 + severity * 0.9          # severity 1 -> 1.9, severity 5 -> 5.5
    linear_scale = 1.0 - severity * 0.12  # extra flat attenuation, severity 5 -> 0.4

    dark = np.clip(img * linear_scale, 0.0, 1.0) ** gamma

    if add_noise:
        # shot + read noise grows as exposure drops -- scale sigma with severity
        sigma = 0.01 + severity * 0.012
        noise = rng.normal(0.0, sigma, size=dark.shape).astype(np.float32)
        dark = np.clip(dark + noise, 0.0, 1.0)

    return dark


def indoor_haze(img: np.ndarray, depth_m: np.ndarray, severity: int = 3,
                 atmospheric_light: float = 0.85) -> np.ndarray:
    """
    Simulate indoor haze/smoke/dust (e.g. warehouse dust, kitchen steam, light
    smoke) using the standard depth-based atmospheric scattering model:

        I(x) = J(x) * t(x) + A * (1 - t(x)),   t(x) = exp(-beta * depth(x))

    This is the same model used for outdoor synthetic fog; only the label and the
    beta range differ here to stay physically plausible at indoor scale (a few
    metres) rather than outdoor scale (tens to hundreds of metres).

    depth_m: ground-truth depth in metres, same (H, W) as img. Required -- indoor
    haze density must vary with distance to be physically meaningful; a
    constant-density fog over a whole indoor frame does not look like real haze.

    severity 1 (light haze near the far wall) .. 5 (thick smoke, near objects
    only faintly visible through it).
    """
    img = _to_float01(img)
    depth_m = np.asarray(depth_m, dtype=np.float32)
    depth_m = np.nan_to_num(depth_m, nan=np.nanmax(depth_m) if np.isfinite(depth_m).any() else 5.0)

    # indoor scale: beta tuned so a 3-5m room produces visible density by severity 3-4
    beta = 0.15 + severity * 0.35   # severity 1 -> 0.5, severity 5 -> 1.9
    transmission = np.exp(-beta * depth_m)
    transmission = transmission[..., None]  # broadcast over RGB channels

    hazy = img * transmission + atmospheric_light * (1.0 - transmission)
    return np.clip(hazy, 0.0, 1.0)


def sensor_noise(img: np.ndarray, severity: int = 3, kind: str = "gaussian",
                  rng: np.random.Generator | None = None) -> np.ndarray:
    """
    Simulate sensor noise independent of lighting (e.g. RealSense IR interference,
    CMOS read noise, JPEG-adjacent compression artefacts on a robot's onboard
    camera feed). kind: "gaussian" or "poisson".
    """
    rng = rng or np.random.default_rng()
    img = _to_float01(img)

    if kind == "poisson":
        # scale controls how "grainy" -- lower scale = fewer photons = more noise
        scale = max(1.0, 60.0 - severity * 10.0)
        noisy = rng.poisson(img * scale) / scale
    else:
        sigma = 0.008 + severity * 0.015
        noisy = img + rng.normal(0.0, sigma, size=img.shape).astype(np.float32)

    return np.clip(noisy, 0.0, 1.0).astype(np.float32)


def motion_or_defocus_blur(img: np.ndarray, severity: int = 3,
                            kind: str = "defocus") -> np.ndarray:
    """
    Simulate blur from robot motion or an out-of-focus RealSense frame during
    driving. Requires OpenCV (imported lazily so this module has no hard
    dependency on it for the other corruptions).
    """
    import cv2

    img = _to_float01(img)
    ksize = 2 * severity + 1  # severity 1 -> 3px, severity 5 -> 11px

    if kind == "motion":
        kernel = np.zeros((ksize, ksize), dtype=np.float32)
        kernel[ksize // 2, :] = 1.0
        kernel /= kernel.sum()
        blurred = cv2.filter2D(img, -1, kernel)
    else:
        blurred = cv2.GaussianBlur(img, (ksize, ksize), sigmaX=severity * 0.6)

    return np.clip(blurred, 0.0, 1.0).astype(np.float32)


CORRUPTIONS = {
    "low_light": low_light,
    "indoor_haze": indoor_haze,      # needs depth_m kwarg
    "sensor_noise": sensor_noise,
    "blur": motion_or_defocus_blur,
}


def apply_corruption(name: str, img: np.ndarray, severity: int = 3, **kwargs) -> np.ndarray:
    """Dispatch helper so scripts/run_eval.py can iterate over CORRUPTIONS by name."""
    if name not in CORRUPTIONS:
        raise ValueError(f"Unknown corruption '{name}'. Options: {list(CORRUPTIONS)}")
    return CORRUPTIONS[name](img, severity=severity, **kwargs)
