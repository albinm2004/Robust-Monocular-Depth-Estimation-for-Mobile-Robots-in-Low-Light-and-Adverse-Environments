# Robust Monocular Depth Estimation for the Sherpa RP — Indoor Phase

Working pipeline for the plan in `docs/research-plan-review.md`: indoor corruption
synthesis → CLAHE/Zero-DCE enhancement → Depth Anything V2 inference → scale-aligned
metrics (incl. the RealSense near-field band) → latency benchmarking.

**This is the `beta` branch.** It adds real ROS 2 / Sherpa RP hardware
integration on top of everything below -- see `docs/ROS2_INTEGRATION.md` and
`ros2_ws/src/depth_estimation_node/`. Not yet run on ROS 2 or the actual robot
(this dev machine has neither); `main` stays the clean, fully-validated offline
research pipeline until this is tested on hardware.

## Status (real GPU workstation, RTX 4060 Laptop, CUDA 12.8) — full pipeline complete

Every stage has now run for real: real NYU Depth V2 data, real Depth Anything V2
weights (small/base/large), real Zero-DCE training, real GPU benchmarking. See
`data/results/results_summary.md` for the final results table.

- **Real dataset.** `scripts/run_eval.py --data-source nyu` loads the official NYU
  Depth V2 labeled test split (654 images) via the parquet-converted mirror of
  `sayakpaul/nyu_depth_v2` on the HF Hub — verified real float32 metric depth in
  metres, same image count as the official split. `--data-source synthetic`
  (default) still works unchanged for quick wiring tests.
- **Real model, all three sizes.** `DepthAnythingV2Model` (metric checkpoints)
  verified end-to-end: small on the full 654-image test split (primary results,
  `eval_full654_small.csv`), base and large on an 80-image subset for the
  accuracy-vs-latency comparison (`eval_subset80_{base,large}.csv`).
- **Fixed a real calibration issue.** The "Metric-Indoor" checkpoint's raw output
  carries a consistent ~20–35% global scale bias against this data source's ground
  truth (correlation ~0.93–0.95, but a systematic mean-ratio offset — not noise).
  `resolve_depth()` in `run_eval.py` applies the median-scale calibration
  `depth_infer.py` already shipped for this case, bringing clean-image Abs Rel from
  ~0.30–0.39 down to ~0.05–0.11 (varies by model size), matching published DA-V2
  numbers, and restoring the expected monotonic error-vs-severity trend (confirmed
  at both 80-image and full-654 scale).
- **Zero-DCE trained for real**: 5000 steps / 100 epochs on 400 low-light images
  (from NYU's *train* split, kept separate from the test split used for eval;
  loss 3.11 → 0.185, weights in `data/results/zero_dce_weights.pth`). Re-ran the
  `zero_dce` eval arm with `--zero-dce-weights` against the full 654-image split.
  **Finding**: trained Zero-DCE only helps on `low_light` (the corruption it was
  effectively trained against); it's neutral-to-harmful on every other corruption
  and even on clean images, because the exposure loss pushes every input toward a
  fixed target brightness regardless of whether it needs it — confirmed by direct
  inspection (clean image mean 0.503→0.699 after enhancement; a severe low-light
  input's local pixel std goes from 0.04→0.41, i.e. the curve-based enhancement
  amplifies sensor noise right along with the signal, since it has no denoising
  term). Visible directly in `data/results/qualitative_low_light_sev*.png`.
- **Dataset limitation found, not silently swallowed**: `near_field_pixel_count`
  is 0 across every row checked (25,506+ rows) — NYU Depth V2 (Kinect-captured)
  has essentially no ground-truth depth in the RealSense's 0.25–0.70 m
  obstacle-detection band. Not a bug (the mask/metric code is unit-tested correct
  on synthetic data that does cover that range) — real near-field validation needs
  actual RealSense captures, per `research-plan-review.md`'s "future work" note.
- **Dev-GPU latency/FPS** (`scripts/benchmark_real_model.py`, dedicated from the
  accuracy sweep to avoid 4x'ing an already multi-hour run): small 52ms median
  (11.7 fps mean), base 107ms (6.5 fps), large 344ms (2.7 fps) on this RTX 4060
  Laptop. Explicitly labelled dev-GPU — Jetson Orin Nano numbers are a separate
  later step per `docs/GPU_JETSON_SETUP.md`, not done here.
- **Qualitative figures** with real predictions in `data/results/` (low_light
  sev3/sev5, indoor_haze sev4) — also surface the indoor_haze finding that depth
  prediction collapses toward near-uniform under heavy haze, since the atmospheric
  scattering model destroys structural contrast no tone-curve enhancement (CLAHE
  or Zero-DCE) can recover.
- **Reliability fix for long real-model runs**: a multi-hour sweep was originally
  silent (Python fully buffers stdout off a terminal) and only wrote its CSV once
  at the end, so a kill/crash lost everything -- confirmed once, the hard way.
  `run_eval.py` now caps thread pools (numpy/OpenCV/torch were each spinning up
  one pool per core), prints per-image progress with an ETA, and writes each
  image's rows incrementally.

To reproduce the primary run:
```bash
python -u scripts/run_eval.py --data-source nyu --model-size small --real-model \
    --skip-benchmark --out-csv data/results/eval_full654_small.csv
```
(shows live per-image progress/ETA; safe to kill and re-run, each image's rows are
flushed as they're computed.)

## Layout

```
src/
  corruptions.py       low_light, indoor_haze, sensor_noise, blur  (indoor-scoped only)
  enhance.py            CLAHE + Zero-DCE (DCE-Net + 4 non-reference losses)
  depth_infer.py        Depth Anything V2 wrapper + scale-alignment math
  metrics.py             Abs Rel / RMSE / delta1 + near-field (0.25-0.70m) band
  benchmark.py            latency/FPS timing harness -> CSV
scripts/
  make_sample_data.py     synthetic indoor RGB+depth pairs (for wiring tests only)
  train_zero_dce.py       unsupervised Zero-DCE training loop
  run_eval.py             full corruption x enhancement x model x metrics sweep
  make_qualitative_figure.py   side-by-side comparison figure for the paper
  benchmark_real_model.py      dedicated dev-GPU latency/FPS across model sizes
  make_results_summary.py      aggregates run_eval.py CSVs into an IEEE-ready table
docs/
  research-plan-review.md   the plan review (copied in for reference)
  GPU_JETSON_SETUP.md        how to run this for real on your workstation / robot
data/
  samples/                 synthetic wiring-test images (make_sample_data.py output)
  zero_dce_train/          400 low-light images (from NYU *train* split) used to train Zero-DCE
  results/
    eval_full654_small.csv                        primary results (real DA-V2 small, full NYU test split)
    eval_full654_small_zerodcetrained*.csv         zero_dce arm re-run with trained weights
    eval_subset80_{small,base,large}.csv           80-image accuracy-vs-latency comparison
    zero_dce_weights.pth                            trained Zero-DCE weights
    benchmark_dev_gpu.csv                           latency/FPS per model size
    results_summary.{md,csv}                        final aggregated results table
    qualitative_*.png                               real-prediction comparison figures
```

## Quick start (works anywhere, no GPU/HF needed — validates the pipeline)

```bash
pip install -r requirements.txt
python scripts/make_sample_data.py
python scripts/run_eval.py --data-dir data/samples
python scripts/make_qualitative_figure.py --corruption indoor_haze --severity 4
```

## Running for real

See `docs/GPU_JETSON_SETUP.md` for GPU workstation / Jetson setup. Short version:

```bash
python -u scripts/run_eval.py --data-source nyu --model-size small --real-model \
    --skip-benchmark --out-csv data/results/eval_full654_small.csv
```

`--data-source nyu` downloads/caches the real 654-image NYU Depth V2 labeled test
split automatically (no manual `.mat` download needed). Drop `--skip-benchmark` if
you want per-row timing folded into the same run instead of benchmarking
separately; add `--limit N` for a cheap subset before committing to the full run;
add `--zero-dce-weights <path>` once you've trained real weights via
`train_zero_dce.py`. `--data-dir`/`--data-source synthetic` (the default) still
works unchanged for quick wiring tests without touching the network.

## Known simplifications to fix before this becomes paper results

1. ~~`data/samples/` is synthetic procedural geometry~~ — **done**: real NYU Depth V2
   test split wired in via `--data-source nyu`; primary results (small checkpoint)
   run on the full 654-image official test split. Synthetic loader remains for
   wiring tests only.
2. ~~Zero-DCE ships untrained~~ — **done**: trained for real (5000 steps on 400
   real low-light NYU-train images, loss 3.11→0.185). **New finding, not a
   simplification but worth flagging in the paper**: trained Zero-DCE only
   improves results on `low_light`; it's neutral-to-harmful on every other
   corruption and on clean images, because it unconditionally pushes every input
   toward a fixed target brightness. See Status above and
   `data/results/qualitative_low_light_sev*.png` for the visual evidence (heavy
   noise amplification on severe low-light inputs).
3. ~~`MockDepthModel`~~ — **done**: real Depth Anything V2 (`--real-model`)
   verified end-to-end for all reported results, including the median-scale
   calibration fix (see Status above). `MockDepthModel` still exists for fast
   wiring tests but is not what any reported number comes from.
4. The near-field metric mask in `metrics.py` uses the RealSense's *configured
   detection range* (0.25-0.70m) from the manual — confirmed correct against the
   manual, but **empirically unreachable on NYU Depth V2** (0 ground-truth pixels
   in that band across all 25,506+ rows checked, since Kinect-captured NYU scenes
   don't get that close to the camera). This is a genuine dataset limitation, not
   a code bug (the mask/metric is unit-tested correct on synthetic data that does
   cover the range) — real near-field numbers need actual RealSense captures on
   the Sherpa RP itself, which is explicitly future work in
   `research-plan-review.md`, not part of this synthetic-corruption benchmark.
5. ~~Full 654-image sweep, base/large comparison, GPU benchmarking, qualitative
   figures~~ — **all done**, see `data/results/results_summary.md` for the
   consolidated table. Remaining future work (not part of this phase): Jetson
   Orin Nano latency numbers (`docs/GPU_JETSON_SETUP.md`) and the small
   real-robot RealSense validation set `research-plan-review.md` recommends as
   the highest-leverage addition beyond this synthetic benchmark.
