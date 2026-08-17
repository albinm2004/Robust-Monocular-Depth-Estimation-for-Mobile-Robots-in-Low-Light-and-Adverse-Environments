# Plan Review: Robust Monocular Depth Estimation for Mobile Robots in Low-Light and Adverse Environments

*Reviewed against: Sherpa RP Operating Manual v1.1 (Ati Robotics) and da Silva et al. 2020, "Monocular Vision Aided Depth Map from RGB Images to Estimate Localization and Support Navigation of Mobile Robots," IEEE Sensors Journal.*

**Scope note (confirmed with user, 2026-08-17):** This phase of the project is indoor-only. An outdoor dataset is planned for a later phase and is treated as future work below, not part of the current 6-week plan. This actually strengthens the fit with the Sherpa RP, which is documented as indoor-only operation.

## Bottom line

The six-week plan is a workable evaluation study, but as written it is a **synthetic-benchmark paper** — corrupt NYU Depth V2, run a pretrained depth model, measure error. That is a saturated area (dozens of "robust monocular depth under fog/rain/night" papers already exist on arXiv). The manual shows you actually have a physical robot with a depth camera and an embedded GPU sitting on it. That is the thing that makes this publishable as a *robotics* contribution rather than a *benchmarking* exercise, and the current plan doesn't use it until the very end (FPS benchmarking only). Below are the specific gaps, the reasoning, and a revised week-by-week plan.

## What the manual tells us that should shape the plan

- **Onboard compute is an NVIDIA Jetson Orin Nano** running ROS 2 Humble on Ubuntu 22.04. This is the real deployment target. "Benchmark FPS/latency for each combination on your GPU" in the original plan is ambiguous — it should explicitly mean the Jetson, not just your dev workstation. Reviewers care about edge-device numbers far more than desktop-GPU numbers for a "mobile robot" claim.
- **The robot already carries an Intel RealSense D435i** (RGB + active stereo depth, 1280×720 @ 30 Hz, mounted 0.10 m above the floor). This camera gives you real RGB frames *and* real depth ground truth simultaneously, on your own hardware, in your own lab's lighting. That is a free source of a small real-world evaluation set — something almost no competing "low-light depth" paper has, because most only test on public datasets.
- **Sherpa RP is indoor-only** ("Operate indoors only, unless otherwise approved"). That's a good match for NYU Depth V2 (indoor) and for the indoor-only scope now confirmed for this phase. It's a weaker match for synthetic **fog**, which is a scattering model designed for outdoor atmospheric haze. Since outdoor is explicitly deferred, this phase should drop "fog" as a label entirely and use the same scattering math relabeled as indoor haze/smoke/steam/dust — keeps the corruption pipeline code identical, just makes the framing honest for an indoor-only paper.
- **The RealSense's configured detection range is 0.25–0.70 m**, used for close-range blind-spot obstacle detection (the Livox LiDAR has a ~0.5 m blind zone at floor level that the RealSense is specifically there to cover). This is a strong hook: you can report depth error *specifically in the near-field blind-spot band*, tying your metric directly to a documented safety-critical function of this robot, not just abstract Abs Rel/RMSE.
- The manual also flags that the LiDAR's built-in IMU is currently unfused due to gyro bias drift, and the robot relies on wheel odometry — not directly relevant to depth estimation, but worth knowing if you later want to fuse depth into navigation; the localization stack (AMCL) is odometry-only right now.

## The one technical correction that matters most

Depth Anything v2's base checkpoints output **relative (scale-and-shift-ambiguous) depth**, not metric depth. If Week 4 computes Abs Rel/RMSE directly against NYU Depth V2's metric ground truth without first aligning scale (the standard per-image median-scaling / least-squares alignment used in MiDaS/Eigen-style evaluation), the numbers will be wrong or meaningless. Two options:

1. Use the **Depth Anything V2 Metric** checkpoint (fine-tuned on NYU/KITTI, outputs metric depth directly) — simplest, and matches indoor NYU-style scenes.
2. Or explicitly apply scale/shift alignment before computing Abs Rel/RMSE/δ1 and say so in the methodology — reviewers will ask if this is missing.

This should be nailed down in Week 1–2, not discovered in Week 4.

## Other gaps worth closing

- **Only one depth model.** A single-model study reads as a case study, not a benchmark. Add one more baseline (e.g., MiDaS v3.1/DPT, or ZoeDepth) so you can report a small comparison table. If compute time is a concern, testing Depth Anything v2's small/base/large variants against each other is a lighter-weight substitute that also gives you an accuracy-vs-latency tradeoff curve — genuinely useful for an edge-deployment paper.
- **No related-work track.** The da Silva et al. 2020 paper you shared is useful background (RGB-D + CNN transfer learning for indoor robot localization, real Kinect hardware, solid methodology section to model your own writing on) but it is not about depth-estimation robustness — it's a localization/classification paper. You'll need actual related work on monocular depth robustness (Depth Anything, ZoeDepth, MiDaS, low-light/adverse-weather depth papers). Start collecting these in Week 1 in parallel, not in Week 5.
- **No real-robot validation step.** This is the biggest missed opportunity. Even a small addition — capture ~50–100 real RGB-D frame pairs from the Sherpa's RealSense in your actual lab under normal and dimmed lighting, run the same pipeline, report how real-world error compares to synthetic-corruption error — turns this from "we corrupted a public dataset" into "we validated on the target robot." That's a defensible, novel contribution for a 6-week project.
- **Writing timeline is backloaded.** Compiling a full IEEE paper (related work, methodology, results, figures, abstract, intro, formatting, references) in the last two weeks alongside finishing experiments is tight. Better to treat the LaTeX doc as a living document from Week 2 onward.
- **Clarify scope: zero-shot evaluation, not training.** Nothing in the plan mentions fine-tuning — confirm this is a zero-shot evaluation study (pretrained Depth Anything v2 + classical enhancement preprocessing). If so, you don't need the full 2.8 GB NYU subset; the official NYU Depth V2 test split (~654 labeled images) plus a small held-out slice for tuning corruption severities is enough, and it's much faster to iterate on.

## Revised plan

**Week 1 — Environment, dataset, and related work (in parallel)**
- Install PyTorch, Hugging Face Transformers, OpenCV, Kornia on both your dev machine and (via SSH) the Jetson Orin Nano.
- Download NYU Depth V2 official test split (~654 images) rather than the full subset, unless you specifically need more scenes for corruption-severity tuning.
- Decide and document: Depth Anything V2 Metric checkpoint (recommended) vs. relative-depth + explicit scale alignment.
- Start a running related-work list: Depth Anything v1/v2, ZoeDepth, MiDaS, any low-light/adverse-weather monocular depth papers, plus da Silva et al. 2020 as an RGB-D/mobile-robot methodology reference.
- Confirm SSH access to the Sherpa RP Jetson (`ssh atiinorbit@<ip>`) and that the RealSense topics (`/camera/color/image_raw`, `/camera/camera/depth/image_rect_raw`) are publishing — this unblocks the real-data step later.

**Week 2 — Corruption pipeline and clean baseline**
- Write the synthetic low-light / haze / noise corruption script, scoped to indoor-plausible conditions only (dim/uneven lighting, indoor haze-smoke-dust, sensor noise/blur). No outdoor-specific corruptions (rain streaks, atmospheric fog, sun glare) this phase — save those for the outdoor dataset phase.
- Establish clean-image baseline scores with the chosen Depth Anything v2 checkpoint, using the correct metric/alignment approach settled in Week 1.
- Add a second baseline model (or a second Depth Anything v2 size) for comparison.

**Week 3 — Enhancement modules + first real-robot capture**
- Add CLAHE and Zero-DCE preprocessing.
- Run inference on degraded vs. enhanced image sets for both models.
- In parallel: capture a small real RGB-D set from the Sherpa RP's RealSense D435i under normal and low-light lab conditions (even a static capture session is enough — you don't need the robot driving).

**Week 4 — Metrics and benchmarking**
- Compute Abs Rel, RMSE, δ1 across all synthetic conditions, both models, with/without enhancement.
- Add a near-field metric restricted to the 0.25–0.70 m band that matches the RealSense's documented obstacle-detection range — ties results to a real safety function of the robot.
- Benchmark FPS/latency on the dev GPU *and* on the Jetson Orin Nano (the actual deployment target — this is the number reviewers will look for first).
- Run the same pipeline on the small real-robot capture set from Week 3 and compare real-world vs. synthetic-corruption error trends.

**Week 5 — Writing: Methodology, Results, related work**
- By now the LaTeX doc should already have a skeleton from Week 1–2; this week is filling in Methodology, Results, and Related Work, plus generating qualitative side-by-side depth-map figures (clean vs. degraded vs. enhanced, and synthetic vs. real capture).

**Week 6 — Abstract, Introduction, polish, advisor review buffer**
- Write Abstract and Introduction last (once results are final).
- Format tables/references, proofread, compile.
- Leave the back half of the week for advisor feedback and revisions rather than treating compilation as the final step — this is usually the actual bottleneck in a 6-week timeline.

## Future work: outdoor phase (not part of this 6-week plan)

Worth a short forward-looking paragraph in the paper's Conclusion/Future Work section rather than any implementation now:

- **Dataset candidates:** KITTI or Cityscapes for driving-style outdoor scenes, or nuScenes if you want adverse-weather-labeled data directly. If the eventual goal is still "robot deployment" rather than autonomous driving, a smaller outdoor robot dataset (e.g., a campus/pathway RGB-D or stereo set) may be a better narrative fit than driving benchmarks.
- **Corruption types become genuinely different outdoors:** real atmospheric fog/haze, rain streaks/droplets on the lens, direct sun glare and hard shadows, motion blur from uneven terrain — none of these are close analogues of the indoor conditions you're testing now, so the corruption pipeline will need a second, outdoor-specific version rather than a simple relabel.
- **Hardware implication:** Sherpa RP is explicitly indoor-only per the manual, so an outdoor phase would need either a different robot/platform or hand-held/vehicle-mounted RealSense captures rather than robot-collected data — worth flagging to your advisor early since it changes what "real-robot validation" can mean for that phase.
- This indoor phase can stand alone as a complete paper; the outdoor extension reads naturally as either a "Future Work" paragraph or the seed of a follow-up submission.

## Suggested question for your advisor

Whether a small real-robot validation pass (Week 3–4) is in scope given the timeline — it's the highest-leverage addition here, but if the 6-week window is hard-fixed and lab access to the Sherpa RP is limited, it's reasonable to scope it down to "future work" and keep the study fully synthetic. Worth a quick check before committing Week 3 time to it.
