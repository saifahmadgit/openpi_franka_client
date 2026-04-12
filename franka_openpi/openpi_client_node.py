import asyncio
import threading

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from openpi_client import websocket_client_policy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState

from franka_openpi.action_executor import ActionExecutor

JOINT_ORDER = [
    "panda_joint1",
    "panda_joint2",
    "panda_joint3",
    "panda_joint4",
    "panda_joint5",
    "panda_joint6",
    "panda_joint7",
    "panda_finger_joint1",
    "panda_finger_joint2",
]

# State layout expected by the server: (9,) float32
# [0:7] = panda_joint1-7
# [7:9] = panda_finger_joint1, panda_finger_joint2
STATE_DIM = 9


class OpenPIClientNode(Node):
    def __init__(self):
        super().__init__("openpi_client")

        # Parameters
        self.declare_parameter("server_host", "192.168.1.X")  # GPU server IP
        self.declare_parameter("server_port", 8000)
        self.declare_parameter("prompt", "pick up the red apple")
        self.declare_parameter("action_horizon", 10)
        self.declare_parameter("num_episodes", 5)

        host = self.get_parameter("server_host").value
        port = self.get_parameter("server_port").value
        self.prompt = self.get_parameter("prompt").value
        self.action_horizon = self.get_parameter("action_horizon").value

        self.bridge = CvBridge()
        self.front_image = None
        self.wrist_image = None
        self._raw_joints = np.zeros(STATE_DIM)  # 7 joints + 2 finger joints

        # Camera subscribers
        self.create_subscription(
            Image, "/camera/front/camera/color/image_raw", self._front_cb, 10
        )
        self.create_subscription(
            Image, "/camera/wrist/camera/color/image_raw", self._wrist_cb, 10
        )

        # Joint state subscriber
        self.create_subscription(JointState, "/joint_states", self._joint_cb, 10)

        # OpenPI server connection
        self.get_logger().info(f"Connecting to openpi server at {host}:{port}")
        self.policy = websocket_client_policy.WebsocketClientPolicy(
            host=host, port=port
        )
        self.get_logger().info("Connected.")

        # Motion executor
        self.action_executor = ActionExecutor(self)

    # ── callbacks ────────────────────────────────────────────────────────
    def _front_cb(self, msg):
        img = self.bridge.imgmsg_to_cv2(msg, "rgb8")             # (480, 640, 3) HWC
        img = np.ascontiguousarray(img[:, 80:560, :])             # center crop → (480, 480, 3)
        img = cv2.resize(img, (512, 512))                         # (512, 512, 3) HWC uint8
        self.front_image = np.ascontiguousarray(img.transpose(2, 0, 1))  # → (3, 512, 512) CHW

    def _wrist_cb(self, msg):
        img = self.bridge.imgmsg_to_cv2(msg, "rgb8")             # (480, 640, 3) HWC
        img = np.ascontiguousarray(img[:, 80:560, :])             # center crop → (480, 480, 3)
        img = cv2.resize(img, (256, 256))                         # (256, 256, 3) HWC uint8
        self.wrist_image = np.ascontiguousarray(img.transpose(2, 0, 1))  # → (3, 256, 256) CHW

    def _joint_cb(self, msg):
        name_to_pos = dict(zip(msg.name, msg.position))
        for i, jname in enumerate(JOINT_ORDER):
            if jname in name_to_pos:
                self._raw_joints[i] = name_to_pos[jname]

    def _build_state(self) -> np.ndarray:
        """Pack robot state into the (9,) format the server expects.

        Layout: [panda_joint1-7, panda_finger_joint1, panda_finger_joint2]
        """
        return self._raw_joints.astype(np.float32)

    # ── observation packing ──────────────────────────────────────────────
    def _get_observation(self) -> dict | None:
        if self.front_image is None or self.wrist_image is None:
            return None
        return {
            "state": self._build_state(),                          # (9,) float32
            "images": {
                "cam_high": self.front_image,        # (3, 512, 512) CHW uint8
                "cam_left_wrist": self.wrist_image,  # (3, 256, 256) CHW uint8
            },
            "prompt": self.prompt,
        }

    # ── main inference loop ──────────────────────────────────────────────
    def run(self, num_episodes: int = 1):
        """
        Spin rclpy in a background thread so ROS futures resolve while the
        asyncio event loop runs the inference + motion loop on the main thread.
        """
        executor = MultiThreadedExecutor()
        executor.add_node(self)
        spin_thread = threading.Thread(target=executor.spin, daemon=True)
        spin_thread.start()

        try:
            asyncio.run(self._run_async(num_episodes))
        finally:
            executor.shutdown(wait=False)

    async def _run_async(self, num_episodes: int):
        # Wait for both cameras (background executor delivers the callbacks)
        self.get_logger().info("Waiting for camera images...")
        deadline = asyncio.get_event_loop().time() + 10.0
        while self.front_image is None or self.wrist_image is None:
            if asyncio.get_event_loop().time() > deadline:
                self.get_logger().error("Cameras not ready after 10 s — check topics")
                return
            await asyncio.sleep(0.1)
        self.get_logger().info("Cameras ready.")

        for ep in range(num_episodes):
            self.get_logger().info(f"Episode {ep + 1}/{num_episodes}")

            for step in range(500):
                obs = self._get_observation()
                if obs is None:
                    await asyncio.sleep(0.05)
                    continue

                # Query server → (action_horizon, 8)
                result = self.policy.infer(obs)
                actions = np.array(result["actions"])  # (action_horizon, 8)

                self.get_logger().info(
                    f"  Step {step}: executing {len(actions)} waypoints"
                )

                # actions: (action_horizon, 8) — [x, y, z, qw, qx, qy, qz, gripper]
                await self.action_executor.execute_eef_poses(actions)


def main():
    rclpy.init()
    node = OpenPIClientNode()
    num_ep = node.get_parameter("num_episodes").value
    node.run(num_episodes=num_ep)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
