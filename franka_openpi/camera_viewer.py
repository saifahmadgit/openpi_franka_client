#!/usr/bin/env python3
"""Standalone 3-camera viewer. Run in a separate terminal while inference is live.

Usage:
    source install/setup.bash
    python3 camera_viewer.py
Press 'q' to quit.
"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

TOPICS = {
    "front_1": "/camera/front_1/camera/color/image_raw",
    "front_2": "/camera/front_2/camera/color/image_raw",
    "wrist":   "/camera/wrist/camera/color/image_raw",
}
DISPLAY_W = 1280  # total width of the tiled window (1280 / 3 ≈ 427 px per camera)
DISPLAY_H = 320


class CameraViewer(Node):
    def __init__(self):
        super().__init__("camera_viewer")
        self.bridge = CvBridge()
        self.frames = {k: None for k in TOPICS}

        for key, topic in TOPICS.items():
            self.create_subscription(
                Image, topic, lambda msg, k=key: self._cb(msg, k), 10
            )

    def _cb(self, msg, key):
        img = self.bridge.imgmsg_to_cv2(msg, "rgb8")
        self.frames[key] = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    def render(self):
        if any(f is None for f in self.frames.values()):
            return
        panel_w = DISPLAY_W // 3
        panels = []
        for k in ["front_1", "front_2", "wrist"]:
            frame = self.frames[k]
            h, w = frame.shape[:2]
            scale = min(panel_w / w, DISPLAY_H / h)
            new_w, new_h = int(w * scale), int(h * scale)
            resized = cv2.resize(frame, (new_w, new_h))
            panel = np.zeros((DISPLAY_H, panel_w, 3), dtype=np.uint8)
            y = (DISPLAY_H - new_h) // 2
            x = (panel_w - new_w) // 2
            panel[y:y + new_h, x:x + new_w] = resized
            panels.append(panel)
        row = np.hstack(panels)
        for i, label in enumerate(["front_1", "front_2", "wrist"]):
            cv2.putText(
                row, label,
                (i * panel_w + 8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
            )
        cv2.imshow("Camera Feeds  [q to quit]", row)


def main():
    rclpy.init()
    node = CameraViewer()
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.01)
            node.render()
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
