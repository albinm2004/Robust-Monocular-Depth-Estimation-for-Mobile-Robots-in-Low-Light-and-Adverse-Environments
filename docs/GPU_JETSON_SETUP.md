# Running for real: GPU workstation and Jetson Orin Nano (Sherpa RP)

Everything in this repo was built and wiring-tested in a cloud sandbox with no
GPU and no route to huggingface.co. The code is complete; it just hasn't touched
real model weights yet. This doc is the two places that first real run needs to
happen.

## 1. Your GPU workstation (development, accuracy numbers)

```bash
git clone <wherever you push this repo>   # or just copy the folder over
cd depth_project
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cu121   # match your CUDA version

# get the real dataset (NYU Depth V2 official test split, ~654 labeled images)
# see notes below — swap data/samples for a real loader

python scripts/run_eval.py --data-dir <path-to-nyu-test-split> --model-size small --real-model
```

First call to `DepthAnythingV2Model.load()` will download the checkpoint from
`huggingface.co` (a few hundred MB for the small model) — this is exactly the step
that fails in the cloud sandbox and needs to happen here instead.

To sweep model sizes for the accuracy-vs-latency comparison recommended in the plan
review, just change `--model-size small|base|large` and re-run — everything else
(corruption, enhancement, metrics, benchmarking) stays identical.

### About the dataset loader

`scripts/run_eval.py`'s `load_samples()` currently reads the synthetic
`*_rgb.png` / `*_depth.npy` pairs from `make_sample_data.py`. Real NYU Depth V2 is
distributed as a single `.mat` file (`nyu_depth_v2_labeled.mat`, ~2.8GB, official
host: `http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/`). Swap `load_samples()`
for a loader that reads that .mat (via `h5py` or `scipy.io.loadmat` depending on
format version) and yields the same `(name, rgb, depth_m)` tuples — nothing else in
the pipeline needs to change since every downstream function only depends on that
tuple shape.

## 2. Jetson Orin Nano on the Sherpa RP (deployment / real latency numbers)

Per the operating manual: SSH in first (`ssh atiinorbit@<robot-ip>`, password
`ati112` unless it's been changed), then work from `~/ros2_ws_livox` or wherever
you set up a Python environment for this project — the manual's existing ROS 2
workspace doesn't need to be touched for this benchmarking work, keep it separate.

```bash
ssh atiinorbit@<robot-ip>
python3 -m venv ~/depth_bench_venv && source ~/depth_bench_venv/bin/activate
pip install -r requirements.txt    # copy this file over first, e.g. via scp
```

Two things specific to the Jetson:

- **Use the NVIDIA-built PyTorch wheel for Jetson**, not the generic PyPI one —
  Jetson Orin Nano needs a JetPack-matched build for CUDA to actually work
  (`pip install torch` from plain PyPI will install a CPU-only wheel that won't use
  the GPU at all, silently). Check NVIDIA's Jetson PyTorch install page for the
  wheel matching your JetPack/L4T version before benchmarking — GPU vs CPU latency
  on the same device will look very different and only the GPU number is the one
  worth reporting.
- **This is the number the paper's "for Mobile Robots" claim rests on.** Run
  `scripts/run_eval.py --real-model` here and keep the `_benchmark.csv` output
  separate from the workstation one (e.g. `--out-csv data/results/jetson_eval.csv`)
  so the two are easy to compare side by side in the results table.

To capture real RealSense frames from the robot itself for the small real-world
validation set suggested in the plan review: the manual documents the color topic
as `/camera/color/image_raw` and depth as `/camera/camera/depth/image_rect_raw`
(disabled by default — enable with `camera_obstacle_detection.enabled: true` in
`edubot_config.yaml`). A simple `ros2 topic echo` / `rosbag record` on those two
topics, or a small subscriber node saving synchronized frame pairs, is enough —
you don't need the robot driving, a static capture session under a couple of
lighting conditions is sufficient for a validation set.
