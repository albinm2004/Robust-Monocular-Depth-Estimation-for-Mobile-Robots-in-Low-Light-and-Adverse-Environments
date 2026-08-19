# ROS 2 / Sherpa RP hardware integration (beta)

This is the `beta` branch. `main` is the offline research pipeline
(corruption synthesis, enhancement, evaluation, results) validated against
real NYU Depth V2 data and a real GPU -- see `README.md` and
`data/results/results_summary.md`. This branch adds the piece `main`
never had: code that actually talks to the robot.

## What's here

`ros2_ws/src/depth_estimation_node/` -- a ROS 2 Humble package. See its
own `README.md` for install/run instructions and known limitations. Short
version: it subscribes to the Sherpa RP's RealSense colour stream,
enhances + runs it through Depth Anything V2 (small, by default -- the
practical choice per this project's own accuracy-vs-latency numbers),
scale-calibrates the result live against the RealSense's own stereo
depth, and publishes it as a fallback/cross-check depth source alongside
the manual's existing `realsense_obstacle_detector`.

## Why a fallback, not a replacement

The Sherpa RP's manual (`Sherpa RP Operating Manual.pdf`, not committed to
this repo -- ask whoever supplied it) documents the RealSense D435i as
active stereo, specifically placed to cover the Livox LiDAR's ~0.5m
floor-level blind zone, with a configured 0.25-0.70m obstacle-detection
band. Active stereo is the right primary system where it works. This
project's entire research contribution is showing where vision degrades
(low light, sensor noise/IR interference, haze/dust) -- exactly the
conditions where active stereo also struggles. A monocular fallback that
degrades gracefully alongside the primary sensor, rather than silently
producing garbage, is the natural hardware application of the `main`
branch's findings. It is explicitly not a safety-critical replacement --
see the package README's "Safety integration" section before wiring its
obstacle flag into anything that controls motion.

## What is NOT done

- **Not run on ROS 2 or the robot.** This dev machine has neither
  installed. The code is syntax-checked and follows standard rclpy
  patterns, but needs an on-robot pass before it's trusted.
- **Not wired into `cmd_vel_safety_gate` or any other safety-critical
  node.** Deliberately left as a decision for on-hardware validation.
- **Jetson latency is still unmeasured** (`main`'s benchmark numbers are
  explicitly dev-GPU-only, not Jetson -- see `docs/GPU_JETSON_SETUP.md`
  and `data/results/benchmark_dev_gpu.csv`). The default `inference_hz:
  5.0` in this package is a starting guess.
- **No real-robot RealSense validation set.** `research-plan-review.md`'s
  suggested highest-leverage addition -- capturing real RGB-D frames off
  this exact robot in this exact lab -- is still outstanding on both
  branches.

## Branch policy

Hardware-integration work belongs here, on `beta`, not on `main`. `main`
stays the clean, fully-validated offline research pipeline; merge to
`main` only once this has actually been run against the robot.
