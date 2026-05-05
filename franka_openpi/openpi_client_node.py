import asyncio
import logging
import time
import threading
from typing import Dict, Tuple

import cv2
import numpy as np
import rclpy
import websockets.sync.client
from cv_bridge import CvBridge
from openpi_client import action_chunk_broker, msgpack_numpy, websocket_client_policy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState

from franka_openpi.action_executor import ActionExecutor


class _NoPingPolicy(websocket_client_policy.WebsocketClientPolicy):
    """WebsocketClientPolicy with keepalive pings disabled.

    The default 20 s ping timeout fires during inference when the model
    takes longer than expected (3 cameras + new joint model).
    """

    def _wait_for_server(self) -> Tuple[websockets.sync.client.ClientConnection, Dict]:
        logging.info(f"Waiting for server at {self._uri}...")
        while True:
            try:
                headers = (
                    {"Authorization": f"Api-Key {self._api_key}"}
                    if self._api_key
                    else None
                )
                conn = websockets.sync.client.connect(
                    self._uri,
                    compression=None,
                    max_size=None,
                    additional_headers=headers,
                    ping_interval=None,  # disable keepalive pings
                    ping_timeout=None,
                )
                metadata = msgpack_numpy.unpackb(conn.recv())
                return conn, metadata
            except ConnectionRefusedError:
                logging.info("Still waiting for server...")
                time.sleep(5)

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

# State layout expected by the server: (9,) float32
# [0:7] = panda_joint1-7
# [7:9] = panda_finger_joint1, panda_finger_joint2
STATE_DIM = 9   # joints read from /joint_states (7 arm + 2 finger)
STATE_OUT = 7   # policy expects only the 7 arm joint angles

# All cameras stream at native resolution; server resize_with_pad handles any input size.


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
        self.front1_image = None
        self.front2_image = None
        self.wrist_image = None
        self._raw_joints = np.zeros(STATE_DIM)  # 7 joints + 2 finger joints

        # Camera subscribers (3 cameras matching dataset: front_1, front_2, wrist)
        self.create_subscription(
            Image, "/camera/front_1/camera/color/image_raw", self._front1_cb, 10
        )
        self.create_subscription(
            Image, "/camera/front_2/camera/color/image_raw", self._front2_cb, 10
        )
        self.create_subscription(
            Image, "/camera/wrist/camera/color/image_raw", self._wrist_cb, 10
        )

        # Joint state subscriber
        self.create_subscription(JointState, "/joint_states", self._joint_cb, 10)

        # OpenPI server connection
        self.get_logger().info(f"Connecting to openpi server at {host}:{port}")
        self.policy = action_chunk_broker.ActionChunkBroker(
            _NoPingPolicy(host=host, port=port),
            action_horizon=self.action_horizon,
        )
        self.get_logger().info("Connected.")

        # Motion executor
        self.action_executor = ActionExecutor(self)

    # ── callbacks ────────────────────────────────────────────────────────
    def _proc_image(self, msg) -> np.ndarray:
        """Decode to CHW uint8. Server resize_with_pad(224, 224) handles any input size."""
        img = self.bridge.imgmsg_to_cv2(msg, "rgb8")
        return np.ascontiguousarray(img.transpose(2, 0, 1))

    def _front1_cb(self, msg):
        self.front1_image = self._proc_image(msg)

    def _front2_cb(self, msg):
        self.front2_image = self._proc_image(msg)

    def _wrist_cb(self, msg):
        self.wrist_image = self._proc_image(msg)

    def _joint_cb(self, msg):
        name_to_pos = dict(zip(msg.name, msg.position))
        for i, jname in enumerate(JOINT_ORDER):
            if jname in name_to_pos:
                self._raw_joints[i] = name_to_pos[jname]

    def _build_state(self) -> np.ndarray:
        # Policy expects (7,) — arm joints only, no finger joints
        return self._raw_joints[:STATE_OUT].astype(np.float32)

    # ── observation packing ──────────────────────────────────────────────
    def _get_observation(self) -> dict | None:
        if self.front1_image is None or self.front2_image is None or self.wrist_image is None:
            return None
        return {
            "state": self._build_state(),                              # (7,) float32
            "images": {
                "cam_high":        self.front1_image,  # (3, 480, 640) CHW uint8 ← front_1
                "cam_left_wrist":  self.front2_image,  # (3, 480, 640) CHW uint8 ← front_2
                "cam_right_wrist": self.wrist_image,   # (3, 480, 640) CHW uint8 ← wrist
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
            executor.shutdown()

    async def _run_async(self, num_episodes: int):
        # Wait for all three cameras (background executor delivers the callbacks)
        self.get_logger().info("Waiting for camera images...")
        deadline = asyncio.get_event_loop().time() + 10.0
        while self.front1_image is None or self.front2_image is None or self.wrist_image is None:
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

                # Broker returns one action at a time; re-queries server every action_horizon steps
                result = self.policy.infer(obs)
                action = np.array(result["actions"])  # (8,)

                self.get_logger().info(f"  Step {step}: action = {action}")

                await self.action_executor.execute_joint_commands(action[np.newaxis])


def main():
    rclpy.init()
    node = OpenPIClientNode()
    num_ep = node.get_parameter("num_episodes").value
    node.run(num_episodes=num_ep)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
