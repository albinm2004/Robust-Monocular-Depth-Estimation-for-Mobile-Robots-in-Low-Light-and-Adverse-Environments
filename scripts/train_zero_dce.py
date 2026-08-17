"""
Unsupervised Zero-DCE training loop -- no paired/clean-reference images needed,
only a folder of low-light images (real Sherpa RP captures, or the synthetic
low_light/indoor_haze corruptions from src/corruptions.py work fine to bootstrap
before you have real captures).

Run on your GPU workstation for anything beyond a handful of images -- this will
be very slow on CPU for a real training set. Not run for real in this sandbox
(no GPU here); only exercised on a tiny 2-image, 3-step run to confirm the loop
is wired correctly.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from enhance import (                                            # noqa: E402
    ZeroDCENet,
    color_constancy_loss,
    exposure_loss,
    illumination_smoothness_loss,
    spatial_consistency_loss,
)


class ImageFolderDataset(Dataset):
    def __init__(self, folder: str, size: int = 256):
        self.paths = sorted(glob.glob(os.path.join(folder, "*.png")) + glob.glob(os.path.join(folder, "*.jpg")))
        self.size = size
        if not self.paths:
            raise ValueError(f"No .png/.jpg images found in {folder}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB").resize((self.size, self.size))
        arr = np.asarray(img).astype(np.float32) / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1)


def train(data_dir: str, epochs: int, batch_size: int, lr: float, out_path: str,
          device: str = "cpu", max_steps: int | None = None) -> None:
    ds = ImageFolderDataset(data_dir)
    dl = DataLoader(ds, batch_size=min(batch_size, len(ds)), shuffle=True)

    model = ZeroDCENet().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    # weights per the Zero-DCE paper's loss combination (reasonable defaults)
    w_spatial, w_exposure, w_color, w_smooth = 1.0, 10.0, 5.0, 200.0

    step = 0
    for epoch in range(epochs):
        for batch in dl:
            batch = batch.to(device)
            enhanced, curves = model(batch)

            loss = (
                w_spatial * spatial_consistency_loss(enhanced, batch)
                + w_exposure * exposure_loss(enhanced)
                + w_color * color_constancy_loss(enhanced)
                + w_smooth * illumination_smoothness_loss(curves)
            )

            opt.zero_grad()
            loss.backward()
            opt.step()

            step += 1
            if step % max(1, (max_steps or 10) // 5) == 0 or step == 1:
                print(f"epoch {epoch} step {step} loss {loss.item():.4f}")

            if max_steps and step >= max_steps:
                break
        if max_steps and step >= max_steps:
            break

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    torch.save(model.state_dict(), out_path)
    print(f"saved weights to {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/samples")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--out", default="data/results/zero_dce_weights.pth")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--max-steps", type=int, default=None,
                     help="Cap total optimizer steps -- use a small number (e.g. 3) "
                          "for a quick wiring smoke-test.")
    args = ap.parse_args()

    train(args.data_dir, args.epochs, args.batch_size, args.lr, args.out,
          device=args.device, max_steps=args.max_steps)
