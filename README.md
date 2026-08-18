# Robust Monocular Depth Estimation for the Sherpa RP — Indoor Phase

Working pipeline for the plan in `docs/research-plan-review.md`: indoor corruption
synthesis → CLAHE/Zero-DCE enhancement → Depth Anything V2 inference → scale-aligned
metrics (incl. the RealSense near-field band) → latency benchmarking.

## Status (updated on a real GPU workstation, RTX 4060 Laptop, CUDA 12.8)

The pipeline has now run for real: real NYU Depth V2 data, real Depth Anything V2
weights, on a real GPU. Concretely, as of this update:

- **Real dataset wired in.** `scripts/run_eval.py --data-source nyu` loads the
  official NYU Depth V2 labeled test split (654 images) via the parquet-converted
  mirror of `sayakpaul/nyu_depth_v2` on the HF Hub — verified to be real float32
  metric depth in metres (not a placeholder), same image count as the official
  split. The old synthetic loader (`--data-source synthetic`, the default) still
  works unchanged for quick wiring tests.
- **Real model verified end-to-end.** `DepthAnythingV2Model` (metric, small
  checkpoint) loads and runs real forward passes on GPU.
- **Found and fixed a real calibration issue.** The "Metric-Indoor" checkpoint's
  raw output carries a consistent ~20–35% global scale bias against this data
  source's ground truth (correlation ~0.93–0.95, but a systematic mean-ratio
  offset — not noise). `resolve_depth()` in `run_eval.py` now applies the
  median-scale calibration `depth_infer.py` already shipped for exactly this case.
  This brought clean-image Abs Rel from ~0.30–0.39 down to ~0.05–0.07, matching
  published DA-V2 numbers, and restored the expected monotonic
  error-vs-corruption-severity trend.
- **Sanity-checked on an 80-image real subset** (`data/results/eval_subset80_small.csv`,
  3,120 rows) before committing to the full run: Abs Rel increases monotonically
  with severity for all four corruptions; CLAHE improves low-light results;
  untrained Zero-DCE barely moves either metric (expected — see simplification #2
  below).
- **Found a real dataset limitation.** `near_field_pixel_count` is 0 across every
  row checked so far — NYU Depth V2 (Kinect-captured) has essentially no
  ground-truth depth in the RealSense's 0.25–0.70 m obstacle-detection band, so
  that metric is structurally unreachable on this dataset. It isn't a bug (the
  mask/metric code is unit-tested and correct on synthetic data that does cover
  that range) — real near-field validation needs actual RealSense captures, per
  `research-plan-review.md`'s "future work" note.
- **Fixed a reliability issue for long real-model runs.** A full 654-image sweep
  was originally silent for hours (Python fully buffers stdout when not attached
  to a terminal) and only wrote its CSV once at the very end, so a kill/crash lost
  everything. `run_eval.py` now caps thread pools (numpy/OpenCV/torch were each
  spinning up one pool per core — confirmed via CPU-time delta, this dominated
  wall-clock far more than actual GPU compute did), prints per-image progress with
  an ETA, and writes each image's rows to CSV incrementally.

Still to run (paused mid-session, resumable): the full 654-image sweep on the
small checkpoint, base/large accuracy-vs-latency comparison, real Zero-DCE
training (400 low-light training images from the NYU *train* split are already
generated in `data/zero_dce_train/`), dedicated GPU benchmarking, qualitative
figures with real predictions, and the final results summary table.

To resume the full run:
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
docs/
  research-plan-review.md   the plan review (copied in for reference)
  GPU_JETSON_SETUP.md        how to run this for real on your workstation / robot
data/
  samples/                 synthetic wiring-test images (make_sample_data.py output)
  zero_dce_train/          400 low-light images (from NYU *train* split) for Zero-DCE training
  results/                  CSV / figure / weight outputs land here
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
   test split now wired in via `--data-source nyu` (see Status above). The
   synthetic loader remains available for wiring tests only.
2. Zero-DCE still ships **untrained** (random init) in the results generated so
   far — training data (400 low-light images from NYU's train split) is prepared
   in `data/zero_dce_train/`, but `train_zero_dce.py` hasn't been run for real yet.
   Run it and re-run the `zero_dce` eval arm with `--zero-dce-weights` before
   reporting those numbers.
3. ~~`MockDepthModel`~~ — **done for the runs reported here**: real
   Depth Anything V2 (`--real-model`) has been verified end-to-end, including a
   necessary calibration fix (see Status above). `MockDepthModel` still exists and
   is still useful for wiring tests, but is no longer what the reported numbers
   come from.
4. The near-field metric mask in `metrics.py` uses the RealSense's *configured
   detection range* (0.25-0.70m) from the manual — confirmed correct against the
   manual, but **empirically unreachable on NYU Depth V2** (0 pixels in that band
   across every row checked so far, since Kinect-captured NYU scenes don't get
   that close to the camera). Real near-field numbers need actual RealSense
   captures, not this synthetic-corruption benchmark.
5. Only the full 654-image sweep on the **small** checkpoint, base/large
   accuracy-vs-latency comparison, dedicated GPU benchmarking, and qualitative
   figures with real predictions are still outstanding — the 80-image subset
   sanity check (step before scaling up) passed with correct trends.
