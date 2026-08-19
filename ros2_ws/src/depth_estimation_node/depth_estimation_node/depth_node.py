"""
ROS 2 node: robust monocular depth estimation for the Sherpa RP.

Subscribes to the RealSense colour stream, runs it through enhancement
(CLAHE by default -- see README for why not Zero-DCE by default) and
Depth Anything V2, scale-corrects the result live against the RealSense's
own stereo depth (falling back to the last confidently-computed scale
factor when stereo confidence drops -- e.g. in low light, the exact
condition this whole project is about), and publishes:

  <depth_out_topic>            sensor_msgs/Image, 32FC1, metres
  <obstacle_out_topic>         std_msgs/Bool  -- advisory near-field flag
  ~/scale_factor                std_msgs/Float32  -- diagnostic
  ~/stereo_valid_fraction       std_msgs/Float32  -- diagnostic; low value
                                 means the node is running on a held/stale
                                 scale factor, i.e. exactly the situation
                                 where this fallback is meant to matter

IMPORTANT: <obstacle_out_topic> is advisory only. It is NOT wired into
cmd_vel_safety_gate or any other safety-critical path by this package --
see README.md "Safety integration" before doing that on real hardware.
"""

from __future__ import annotations

import time

import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Float32

from depth_estimation_node.inference import (
    DepthAnythingV2Model,
    ZeroDCENet,
    clahe_enhance,
    median_scale_alignment,
    zero_dce_enhance,
)


class DepthEstimationNode(Node):
    def __init__(self):
        super().__init__('depth_estimation_node')

        self._declare_params()
        p = self._read_params()

        self.bridge = CvBridge()
        self._latest_depth_m: np.ndarray | None = None
        self._scale_factor = 1.0
        self._last_inference_t = 0.0

        self.get_logger().info(
            f"Loading Depth Anything V2 ({p['model_size']}, device={p['device']})...")
        self.model = DepthAnythingV2Model(size=p['model_size'], device=p['device'])
        self.model.load()
        self.get_logger().info("Model loaded.")

        self.zero_dce_net = None
        if p['enhancement'] == 'zero_dce':
            self.zero_dce_net = ZeroDCENet()
            if p['zero_dce_weights_path']:
                self.zero_dce_net.load_weights(p['zero_dce_weights_path'])
                self.get_logger().info(f"Loaded Zero-DCE weights: {p['zero_dce_weights_path']}")
            else:
                self.get_logger().warn(
                    "enhancement=zero_dce but zero_dce_weights_path is empty -- "
                    "using an UNTRAINED net, which per this project's own findings "
                    "barely helps and can hurt. Set zero_dce_weights_path, or use "
                    "enhancement=clahe (the recommended default).")

        self.depth_pub = self.create_publisher(Image, p['depth_out_topic'], 5)
        self.obstacle_pub = self.create_publisher(Bool, p['obstacle_out_topic'], 5)
        self.scale_pub = self.create_publisher(Float32, '~/scale_factor', 5)
        self.valid_frac_pub = self.create_publisher(Float32, '~/stereo_valid_fraction', 5)

        self.create_subscription(Image, p['depth_topic'], self._depth_cb, 5)
        self.create_subscription(Image, p['color_topic'], self._color_cb, 5)

        self.get_logger().info(
            f"Subscribed color={p['color_topic']} depth={p['depth_topic']}, "
            f"publishing depth={p['depth_out_topic']} obstacle={p['obstacle_out_topic']} "
            f"at up to {p['inference_hz']} Hz.")

    # -- parameters -----------------------------------------------------

    def _declare_params(self):
        self.declare_parameter('color_topic', '/camera/color/image_raw')
        self.declare_parameter('depth_topic', '/camera/camera/depth/image_rect_raw')
        self.declare_parameter('depth_out_topic', '/depth_estimation/depth')
        self.declare_parameter('obstacle_out_topic', '/depth_estimation/near_field_obstacle')
        self.declare_parameter('model_size', 'small')  # small|base|large -- see results_summary.md
        self.declare_parameter('device', 'cuda')
        self.declare_parameter('enhancement', 'clahe')  # none|clahe|zero_dce
        self.declare_parameter('zero_dce_weights_path', '')
        self.declare_parameter('inference_hz', 5.0)
        self.declare_parameter('near_field_low_m', 0.25)   # matches the RealSense's
        self.declare_parameter('near_field_high_m', 0.70)  # configured detection range
        self.declare_parameter('obstacle_pixel_threshold', 500)
        self.declare_parameter('stereo_valid_fraction_threshold', 0.3)
        self.declare_parameter('scale_ema_alpha', 0.2)
        self.declare_parameter('depth_min_valid_m', 0.1)
        self.declare_parameter('depth_max_valid_m', 10.0)

    def _read_params(self) -> dict:
        g = self.get_parameter
        return {
            'color_topic': g('color_topic').value,
            'depth_topic': g('depth_topic').value,
            'depth_out_topic': g('depth_out_topic').value,
            'obstacle_out_topic': g('obstacle_out_topic').value,
            'model_size': g('model_size').value,
            'device': g('device').value,
            'enhancement': g('enhancement').value,
            'zero_dce_weights_path': g('zero_dce_weights_path').value,
            'inference_hz': g('inference_hz').value,
        }

    # -- callbacks --------------------------------------------------------

    def _depth_cb(self, msg: Image):
        try:
            raw = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        except Exception as e:  # noqa: BLE001 -- a bad frame must not crash the node
            self.get_logger().warn(f"Failed to decode depth frame: {e}")
            return

        if raw.dtype == np.uint16:
            depth_m = raw.astype(np.float32) / 1000.0  # RealSense raw driver: mm -> m
        else:
            depth_m = raw.astype(np.float32)  # already metres (e.g. rectified/aligned topic)

        self._latest_depth_m = depth_m

    def _color_cb(self, msg: Image):
        inference_hz = self.get_parameter('inference_hz').value
        now = time.monotonic()
        if now - self._last_inference_t < (1.0 / max(inference_hz, 0.1)):
            return  # throttle: DA-V2 forward passes are far slower than the 30Hz camera feed
        self._last_inference_t = now

        try:
            rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
        except Exception as e:  # noqa: BLE001
            self.get_logger().warn(f"Failed to decode color frame: {e}")
            return

        try:
            self._process_frame(rgb, msg.header)
        except Exception as e:  # noqa: BLE001 -- keep the node alive across a bad inference
            self.get_logger().error(f"Inference failed on this frame, skipping: {e}")

    # -- core pipeline ------------------------------------------------------

    def _process_frame(self, rgb_uint8: np.ndarray, header):
        from PIL import Image as PILImage

        p = self._read_params()
        rgb = rgb_uint8.astype(np.float32) / 255.0

        if p['enhancement'] == 'clahe':
            processed = clahe_enhance(rgb)
        elif p['enhancement'] == 'zero_dce' and self.zero_dce_net is not None:
            processed = zero_dce_enhance(rgb, self.zero_dce_net)
        else:
            processed = rgb

        pil_img = PILImage.fromarray((processed * 255).astype(np.uint8))
        pred = self.model.predict(pil_img)

        depth_m = self._latest_depth_m
        low = self.get_parameter('depth_min_valid_m').value
        high = self.get_parameter('depth_max_valid_m').value
        ema_alpha = self.get_parameter('scale_ema_alpha').value
        stereo_thresh = self.get_parameter('stereo_valid_fraction_threshold').value

        valid_frac = 0.0
        if depth_m is not None and depth_m.shape == pred.raw.shape:
            valid_mask = np.isfinite(depth_m) & (depth_m > low) & (depth_m < high)
            valid_frac = float(valid_mask.mean())
            if valid_frac >= stereo_thresh:
                # Enough confident stereo returns this frame -- update the scale
                # factor (EMA-smoothed, not a hard snap, to avoid frame-to-frame jitter).
                _, fresh_scale = median_scale_alignment(pred.raw, depth_m, valid_mask)
                self._scale_factor = (1 - ema_alpha) * self._scale_factor + ema_alpha * fresh_scale
            # else: hold the last known scale factor -- this is the fallback path.
        elif depth_m is not None:
            self.get_logger().warn(
                f"Depth topic shape {depth_m.shape} != predicted depth shape "
                f"{pred.raw.shape} -- scale calibration disabled, holding last "
                f"known scale factor. Check color/depth stream resolutions match.",
                once=True)

        corrected = (pred.raw.astype(np.float64) * self._scale_factor).astype(np.float32)

        depth_msg = self.bridge.cv2_to_imgmsg(corrected, encoding='32FC1')
        depth_msg.header = header
        self.depth_pub.publish(depth_msg)

        near_low = self.get_parameter('near_field_low_m').value
        near_high = self.get_parameter('near_field_high_m').value
        pixel_thresh = self.get_parameter('obstacle_pixel_threshold').value
        near_field_mask = (corrected >= near_low) & (corrected <= near_high)
        obstacle = bool(near_field_mask.sum() > pixel_thresh)
        self.obstacle_pub.publish(Bool(data=obstacle))

        self.scale_pub.publish(Float32(data=float(self._scale_factor)))
        self.valid_frac_pub.publish(Float32(data=valid_frac))


def main(args=None):
    rclpy.init(args=args)
    node = DepthEstimationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
