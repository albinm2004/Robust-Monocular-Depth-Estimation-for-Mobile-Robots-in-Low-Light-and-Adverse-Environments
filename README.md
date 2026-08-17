# Robust Monocular Depth Estimation for the Sherpa RP — Indoor Phase

Working pipeline for the plan in `docs/research-plan-review.md`: indoor corruption
synthesis → CLAHE/Zero-DCE enhancement → Depth Anything V2 inference → scale-aligned
metrics (incl. the RealSense near-field band) → latency benchmarking.

## Status

Every module below has been built and self-tested end-to-end in this cloud sandbox,
**using a mock depth model**, because this sandbox has no network route to
`huggingface.co` (confirmed 403 on the API, resolve, and CDN endpoints — PyPI and
GitHub are reachable, HF specifically is not, likely a network allowlist on this
container). That means:

- All the plumbing — corruption pipeline, CLAHE, Zero-DCE (architecture, losses,
  training loop), scale-alignment math, metrics, benchmarking, CSV/figure output —
  is verified correct and ready to use.
- The actual Depth Anything V2 forward pass has **not** been run anywhere yet. The
  first real run has to happen on a machine with normal internet access: your GPU
  workstation, or the Jetson Orin Nano on the Sherpa RP itself.

Nothing here is faked or assumed — the mock model's numbers are explicitly
mock-model numbers (see `scripts/run_eval.py`'s `MockDepthModel`), and every
self-test prints its own pass/fail so you can re-verify anything by re-running it.

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
  samples/                 synthetic wiring-test images (replace with real NYU Depth V2)
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

See `docs/GPU_JETSON_SETUP.md`. Short version: same commands, add `--real-model`,
run on a machine with internet access, point `--data-dir` at your real NYU Depth V2
export instead of `data/samples`.

## Known simplifications to fix before this becomes paper results

1. `data/samples/` is synthetic procedural geometry, not real photos — only used to
   prove the code paths work. Swap in the real NYU Depth V2 test split.
2. Zero-DCE ships **untrained** (random init) until you run `train_zero_dce.py` for
   real (more than the 2-3 smoke-test steps used here) on a real image set.
3. `MockDepthModel` predicts by adding noise to ground truth, so its output is
   insensitive to the actual corrupted image content — it exists purely to validate
   wiring. Numbers from it must never be reported or compared as if they were real
   results; swap to `--real-model` before drawing any conclusions.
4. The near-field metric mask in `metrics.py` uses the RealSense's *configured
   detection range* (0.25-0.70m) from the manual — confirm this still matches your
   `edubot_config.yaml` if it's been changed from the shipped default.
