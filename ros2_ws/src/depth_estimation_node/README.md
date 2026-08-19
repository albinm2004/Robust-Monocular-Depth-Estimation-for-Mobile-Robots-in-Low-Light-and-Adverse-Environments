# depth_estimation_node

ROS 2 Humble package wrapping this project's robust monocular depth
pipeline (Depth Anything V2 + CLAHE/Zero-DCE enhancement) for the Sherpa
RP. Publishes a monocular depth estimate that is live-calibrated against
the RealSense D435i's own stereo depth, and an advisory near-field
obstacle flag for the RealSense's 0.25-0.70m detection band.

**Status: written and syntax-checked, not yet run on ROS 2 or the actual
robot.** This dev machine has no ROS 2 Humble install and no access to the
Sherpa RP hardware -- the code follows standard rclpy/ament_python
patterns and reuses the exact model/enhancement logic already validated
in `src/depth_infer.py` and `src/enhance.py` against real data (see the
main README and `data/results/`), but the ROS-specific plumbing (topics,
message conversion, timing) needs an on-robot test pass before you trust
it in the loop. Treat this as a strong first draft, not a finished
integration.

## Why this exists

The Sherpa RP already has a working depth pipeline: the RealSense D435i's
active stereo, covering the Livox LiDAR's ~0.5m floor-level blind zone
(`realsense_obstacle_detector`, per the operating manual). That's the
right primary system -- active stereo is generally more reliable than
monocular estimation when it works.

This node is a **fallback/cross-check for when it doesn't**: active
stereo degrades under exactly the conditions this whole project studies
(low light, IR interference/sensor noise, haze/dust). The node:

1. Runs the RGB stream through enhancement + Depth Anything V2.
2. Continuously scale-calibrates that monocular estimate against the
   RealSense's *own* stereo depth, but only when enough of the frame has
   confident stereo returns (`stereo_valid_fraction_threshold`).
3. When stereo confidence drops, holds the last confidently-computed
   scale factor and keeps producing a depth estimate from the monocular
   model alone -- i.e. it's designed to degrade gracefully exactly when
   the primary sensor does.

## Install on the robot

```bash
# On the Jetson, inside the Docker container (see the Sherpa RP manual's
# "Robot Startup and Operation Procedure" for getting into the container)
cd /home/ros/ros2_ws/src
# copy or git-clone this ros2_ws/src/depth_estimation_node/ directory here

# Python deps -- same venv/interpreter ROS 2 uses in the container.
# Use the JetPack-matched torch wheel, NOT plain `pip install torch`
# (see ../../docs/GPU_JETSON_SETUP.md in the main repo -- plain PyPI torch
# on Jetson silently installs a CPU-only build).
pip install transformers opencv-python-headless pillow numpy

cd /home/ros/ros2_ws
colcon build --packages-select depth_estimation_node --symlink-install
source install/setup.bash
```

If you trained real Zero-DCE weights (`data/results/zero_dce_weights.pth`
in the main repo), copy that file to the robot too and point
`zero_dce_weights_path` at it -- though per this project's own findings,
CLAHE (the default) is the safer general-purpose choice; see the
parameter comments in `config/depth_estimation_params.yaml`.

## Run

```bash
# Camera must be enabled first -- it's disabled by default per the manual:
#   camera_obstacle_detection.enabled: true   in edubot_config.yaml

ros2 launch depth_estimation_node depth_estimation_node.launch.py
```

Check it's working:

```bash
ros2 topic hz /depth_estimation/depth
ros2 topic echo /depth_estimation/near_field_obstacle
ros2 topic echo /depth_estimation_node/scale_factor
ros2 topic echo /depth_estimation_node/stereo_valid_fraction   # low value = running on a held/stale scale factor
```

## Safety integration

**`/depth_estimation/near_field_obstacle` is published as an advisory
topic only.** This package does not modify `cmd_vel_safety_gate` or any
other safety-critical node, and you should not wire it in without
on-hardware validation first -- a monocular estimate that's wrong in a
new way real stereo isn't is a worse failure mode than "no coverage,"
because it can give false confidence. If you do want to fuse it in later,
the natural approach (mirroring how `/realsense/obstacle_detected`
already gates the safety layer per the manual's Data Flow & Safety Gate
section) is an AND/OR combination in `cmd_vel_safety_gate`, decided
deliberately and tested on the robot, not a change made from this
codebase alone.

## Integrating with edubot_bringup

This package ships its own launch file so it can be started and stopped
independently while you're validating it. To have it launch automatically
with navigation mode, add a `Node(...)` action for it (see
`launch/depth_estimation_node.launch.py` for the exact parameters) into
`edubot_bringup`'s `nav2_launch.launch.py`, and optionally promote the
parameters you want robot-wide into `edubot_config.yaml`'s override chain
-- this package's own `config/depth_estimation_params.yaml` does not
participate in that chain on its own.

## Known limitations / things to verify on-robot

- **RGB/depth spatial alignment**: the node assumes the color and depth
  streams are pixel-aligned (same resolution, `image_rect_raw` topic
  implies rectified). If your RealSense config doesn't align them, the
  scale-calibration step will silently disable itself (logged once) --
  check `stereo_valid_fraction` reads reasonable values (not near 0)
  under normal lighting to confirm alignment is working.
- **Inference rate on the actual Jetson is unmeasured.** The dev-GPU
  benchmark (RTX 4060 Laptop, `data/results/benchmark_dev_gpu.csv`) is
  explicitly not a Jetson number -- the default `inference_hz: 5.0` is a
  starting guess, not a validated figure. Benchmark on the robot and
  adjust.
- **cv_bridge/message type assumptions** (`rgb8` for color, `passthrough`
  auto-detecting `16UC1` mm vs `32FC1` m for depth) match the RealSense
  ROS driver's typical output but aren't verified against this specific
  robot's actual topic encodings -- check `ros2 topic echo
  /camera/color/image_raw --no-arr` and the depth equivalent for the
  `encoding` field before trusting this.
- No automated tests (`ament_flake8`/`ament_pep257` test deps are
  declared in `package.xml` but no test files are included) -- add them
  if this becomes a long-lived part of the stack.
