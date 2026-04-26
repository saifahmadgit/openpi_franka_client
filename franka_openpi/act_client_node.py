import asyncio
import base64
import threading

import cv2
import numpy as np
import requests
import rclpy
from cv_bridge import CvBridge
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState

from franka_openpi.action_executor import ActionExecutor

JOINT_ORDER = [
    "fer_joint1",
    "fer_joint2",
    "fer_joint3",
    "fer_joint4",
    "fer_joint5",
    "fer_joint6",
    "fer_joint7",
    "fer_finger_joint1",
    "fer_finger_joint2",
]

STATE_DIM = 9   # 7 arm + 2 finger joints read from /joint_states
STATE_OUT = 7   # only arm joints sent in state
IMG_H, IMG_W = 480, 640


def _encode_image(chw_uint8: np.ndarray) -> str:
    """CHW uint8 RGB → HWC → PNG bytes → base64 string."""
    hwc = chw_uint8.transpose(1, 2, 0)  # still RGB — PIL reads PNG as RGB
    ok, buf = cv2.imencode(".png", hwc)
    if not ok:
        raise RuntimeError("PNG encode failed")
    return base64.b64encode(buf.tobytes()).decode("utf-8")


class ACTClientNode(Node):
    def __init__(self):
        super().__init__("act_client")

        self.declare_parameter("server_host", "192.168.1.X")
        self.declare_parameter("server_port", 8001)
        self.declare_parameter("num_episodes", 5)

        host = self.get_parameter("server_host").value
        port = self.get_parameter("server_port").value
        self.infer_url = f"http://{host}:{port}/infer"
        self.reset_url = f"http://{host}:{port}/reset"

        self.bridge = CvBridge()
        self.front1_image = None
        self.front2_image = None
        self.wrist_image = None
        self._raw_joints = np.zeros(STATE_DIM)

        self.create_subscription(
            Image, "/camera/front_1/camera/color/image_raw", self._front1_cb, 10
        )
        self.create_subscription(
            Image, "/camera/front_2/camera/color/image_raw", self._front2_cb, 10
        )
        self.create_subscription(
            Image, "/camera/wrist/camera/color/image_raw", self._wrist_cb, 10
        )
        self.create_subscription(JointState, "/joint_states", self._joint_cb, 10)

        self.action_executor = ActionExecutor(self)
        self.get_logger().info(f"ACT client ready → {self.infer_url}")

    def _proc_image(self, msg) -> np.ndarray:
        img = self.bridge.imgmsg_to_cv2(msg, "rgb8")
        img = cv2.resize(img, (IMG_W, IMG_H))
        return np.ascontiguousarray(img.transpose(2, 0, 1))  # (3,480,640) CHW uint8

    def _front1_cb(self, msg): self.front1_image = self._proc_image(msg)
    def _front2_cb(self, msg): self.front2_image = self._proc_image(msg)
    def _wrist_cb(self, msg):  self.wrist_image  = self._proc_image(msg)

    def _joint_cb(self, msg):
        name_to_pos = dict(zip(msg.name, msg.position))
        for i, jname in enumerate(JOINT_ORDER):
            if jname in name_to_pos:
                self._raw_joints[i] = name_to_pos[jname]

    def _build_payload(self) -> dict | None:
        if self.front1_image is None or self.front2_image is None or self.wrist_image is None:
            return None
        return {
            "state": self._raw_joints[:STATE_OUT].tolist(),
            "images": {
                "cam_high":        _encode_image(self.front1_image),
                "cam_left_wrist":  _encode_image(self.front2_image),
                "cam_right_wrist": _encode_image(self.wrist_image),
            },
        }

    def _post_reset(self):
        try:
            requests.post(self.reset_url, timeout=5)
            self.get_logger().info("Reset sent to ACT server.")
        except Exception as e:
            self.get_logger().warn(f"Reset request failed: {e}")

    def run(self, num_episodes: int = 1):
        executor = MultiThreadedExecutor()
        executor.add_node(self)
        spin_thread = threading.Thread(target=executor.spin, daemon=True)
        spin_thread.start()
        try:
            asyncio.run(self._run_async(num_episodes))
        finally:
            executor.shutdown()

    async def _run_async(self, num_episodes: int):
        self.get_logger().info("Waiting for camera images...")
        deadline = asyncio.get_event_loop().time() + 10.0
        while self.front1_image is None or self.front2_image is None or self.wrist_image is None:
            if asyncio.get_event_loop().time() > deadline:
                self.get_logger().error("Cameras not ready after 10 s — check topics")
                return
            await asyncio.sleep(0.1)
        self.get_logger().info("Cameras ready.")

        loop = asyncio.get_event_loop()

        for ep in range(num_episodes):
            self.get_logger().info(f"Episode {ep + 1}/{num_episodes}")
            self._post_reset()

            for step in range(500):
                payload = self._build_payload()
                if payload is None:
                    await asyncio.sleep(0.05)
                    continue

                try:
                    resp = await loop.run_in_executor(
                        None,
                        lambda p=payload: requests.post(self.infer_url, json=p, timeout=30),
                    )
                    resp.raise_for_status()
                    result = resp.json()
                except Exception as e:
                    self.get_logger().error(f"Infer request failed: {e}")
                    continue

                actions = np.array(result["actions"])  # (1, 8)
                self.get_logger().info(f"  Step {step}: action = {actions[-1]}")
                await self.action_executor.execute_joint_commands(actions)


def main():
    rclpy.init()
    node = ACTClientNode()
    num_ep = node.get_parameter("num_episodes").value
    node.run(num_episodes=num_ep)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
