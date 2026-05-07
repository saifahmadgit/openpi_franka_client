import asyncio
import base64
import threading

import cv2
import matplotlib.pyplot as plt
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

STATE_DIM = 9
STATE_OUT = 7
IMG_H, IMG_W = 480, 640


def _encode_image(chw_uint8: np.ndarray) -> str:
    hwc = chw_uint8.transpose(1, 2, 0)
    ok, buf = cv2.imencode(".png", hwc)
    if not ok:
        raise RuntimeError("PNG encode failed")
    return base64.b64encode(buf.tobytes()).decode("utf-8")


class ACTClientNode(Node):
    def __init__(self):
        super().__init__("act_client")

        self.declare_parameter("server_host", "192.168.1.X")
        self.declare_parameter("server_port", 8001)
        self.declare_parameter("action_horizon", 10)
        self.declare_parameter("num_episodes", 5)

        host = self.get_parameter("server_host").value
        port = self.get_parameter("server_port").value
        self.action_horizon = self.get_parameter("action_horizon").value

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
        return np.ascontiguousarray(img.transpose(2, 0, 1))

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

    def _fetch_chunk(self, loop) -> np.ndarray | None:
        payload = self._build_payload()
        if payload is None:
            return None
        try:
            resp = requests.post(self.infer_url, json=payload, timeout=30)
            resp.raise_for_status()
            return np.array(resp.json()["actions"])  # (100, 8)
        except Exception as e:
            self.get_logger().error(f"Infer request failed: {e}")
            return None

    def _post_reset(self):
        try:
            requests.post(self.reset_url, timeout=5)
            self.get_logger().info("Reset sent to ACT server.")
        except Exception as e:
            self.get_logger().warn(f"Reset request failed: {e}")

    def _plot_debug(self, ep: int, commanded: list, actual: list):
        if not commanded or not actual:
            return
        n = min(len(commanded), len(actual))
        cmd = np.array(commanded[:n])   # (n, 7)
        act = np.array(actual[:n])      # (n, 7)
        chunks = np.arange(n)

        joint_labels = [f"j{i+1}" for i in range(7)]
        fig, axes = plt.subplots(7, 1, figsize=(12, 14), sharex=True)
        fig.suptitle(f"Episode {ep} — Commanded (chunk[-1]) vs Actual (after chunk)", fontsize=12)

        for j, ax in enumerate(axes):
            ax.plot(chunks, cmd[:, j], "b-o", markersize=3, label="commanded")
            ax.plot(chunks, act[:, j], "r-o", markersize=3, label="actual")
            ax.set_ylabel(joint_labels[j] + " (rad)")
            ax.legend(loc="upper right", fontsize=7)
            ax.grid(True, alpha=0.3)

        axes[-1].set_xlabel("chunk index")
        plt.tight_layout()
        plt.savefig(f"/tmp/act_debug_ep{ep}.png", dpi=120)
        self.get_logger().info(f"Debug plot saved → /tmp/act_debug_ep{ep}.png")
        plt.show()

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

            chunk = None
            # per-chunk debug: (chunk_idx, joint) → value
            commanded_log = []   # chunk[action_horizon-1][:7] at query time
            actual_log    = []   # _raw_joints[:7] after chunk completes

            for step in range(500):
                if step % self.action_horizon == 0:
                    new_chunk = await loop.run_in_executor(
                        None, lambda: self._fetch_chunk(loop)
                    )
                    if new_chunk is not None:
                        chunk = new_chunk
                        # record last commanded action of this chunk
                        commanded_log.append(chunk[self.action_horizon - 1][:7].copy())
                    elif chunk is None:
                        self.get_logger().error("No chunk available, skipping step.")
                        continue

                action = chunk[step % self.action_horizon]  # (8,)
                self.get_logger().info(f"  Step {step}: action = {action}")
                await self.action_executor.execute_joint_commands(action[np.newaxis])

                # after last step of a chunk, record actual robot state
                if (step + 1) % self.action_horizon == 0:
                    actual_log.append(self._raw_joints[:7].copy())

            self._plot_debug(ep + 1, commanded_log, actual_log)


def main():
    rclpy.init()
    node = ACTClientNode()
    num_ep = node.get_parameter("num_episodes").value
    node.run(num_episodes=num_ep)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
