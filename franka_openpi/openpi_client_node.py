import asyncio
import logging
import os
import time
import threading
from typing import Dict, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rclpy
import websockets.sync.client
from cv_bridge import CvBridge
from openpi_client import msgpack_numpy, websocket_client_policy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState

from franka_openpi.action_executor import ActionExecutor, STEP_DURATION


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

STATE_DIM = 9   # joints read from /joint_states (7 arm + 2 finger)
STATE_OUT = 8   # policy expects 7 arm joints + 1 gripper (fer_finger_joint1)


class OpenPIClientNode(Node):
    def __init__(self):
        super().__init__("openpi_client")

        self.declare_parameter("server_host", "192.168.1.X")
        self.declare_parameter("server_port", 8000)
        self.declare_parameter("prompt", "pick up the orange cylinder")
        self.declare_parameter("num_episodes", 5)
        self.declare_parameter("exec_horizon", 0)  # 0 = execute full chunk

        host = self.get_parameter("server_host").value
        port = self.get_parameter("server_port").value
        self.prompt = self.get_parameter("prompt").value
        self.exec_horizon = self.get_parameter("exec_horizon").value

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

        self.get_logger().info(f"Connecting to openpi server at {host}:{port}")
        self.policy = _NoPingPolicy(host=host, port=port)
        self.get_logger().info("Connected.")

        self.action_executor = ActionExecutor(self)
        self.commanded_log = []
        self.actual_log = []
        self.chunk_sizes_log = []
        self._debug_dir = os.path.expanduser("~/Franka/src/franka_openpi/debug")
        os.makedirs(self._debug_dir, exist_ok=True)

    # ── callbacks ────────────────────────────────────────────────────────

    def _proc_image(self, msg) -> np.ndarray:
        img = self.bridge.imgmsg_to_cv2(msg, "rgb8")
        return np.ascontiguousarray(img.transpose(2, 0, 1))  # CHW (3, 480, 640)

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

    # ── observation / inference ──────────────────────────────────────────

    def _get_observation(self) -> dict | None:
        if self.front1_image is None or self.front2_image is None or self.wrist_image is None:
            return None
        return {
            "state": self._raw_joints[:STATE_OUT].astype(np.float32),
            "images": {
                "cam_high":        self.front1_image,
                "cam_left_wrist":  self.front2_image,
                "cam_right_wrist": self.wrist_image,
            },
            "prompt": self.prompt,
        }

    def _fetch_chunk_sync(self) -> np.ndarray | None:
        obs = self._get_observation()
        if obs is None:
            return None
        try:
            result = self.policy.infer(obs)
            return np.array(result["actions"])  # (chunk_size, 8) — Pi0.5 default: 50
        except Exception as e:
            self.get_logger().error(f"Infer request failed: {e}")
            return None

    # ── debug logging / plotting ─────────────────────────────────────────

    def _init_episode_logs(self):
        for fname in ("commanded.npy", "actual.npy", "chunks.npy", "openpi_debug.png"):
            p = os.path.join(self._debug_dir, fname)
            if os.path.exists(p):
                os.remove(p)
        self.commanded_log.clear()
        self.actual_log.clear()
        self.chunk_sizes_log.clear()

    def _save_logs(self):
        np.save(os.path.join(self._debug_dir, "commanded.npy"), np.array(self.commanded_log))
        np.save(os.path.join(self._debug_dir, "actual.npy"),    np.array(self.actual_log))
        np.save(os.path.join(self._debug_dir, "chunks.npy"),    np.array(self.chunk_sizes_log))
        if len(self.commanded_log) % 50 == 0:
            self._plot_debug()

    def _plot_debug(self):
        try:
            if not self.commanded_log or not self.actual_log:
                return
            cmd = np.array(self.commanded_log)
            act = np.array(self.actual_log)
            n = min(len(cmd), len(act))
            if n == 0:
                return
            cmd, act = cmd[:n], act[:n]
            steps = np.arange(n)

            fig, axes = plt.subplots(7, 1, figsize=(16, 18), sharex=True)
            fig.suptitle(f"Policy Command vs Robot State — {n} steps", fontsize=13)
            for j, ax in enumerate(axes):
                ax.scatter(steps, cmd[:, j], color="steelblue", s=10, label="commanded (policy)")
                ax.scatter(steps, act[:, j], color="tomato",    s=10, label="actual (robot state)")
                ax.set_ylabel(f"Joint {j + 1}\n(rad)", fontsize=8)
                ax.legend(loc="upper right", fontsize=7)
                ax.grid(True, alpha=0.3)
            axes[-1].set_xlabel("step")
            plt.tight_layout()
            path = os.path.join(self._debug_dir, "openpi_debug.png")
            plt.savefig(path, dpi=150)
            plt.close(fig)
            print(f"[OpenPI] Debug plot saved → {path}")
        except Exception as e:
            print(f"[OpenPI] Plot failed: {e}")

    async def _sample_actual(self, n: int) -> list:
        """Sample actual joint state once per policy step during chunk execution."""
        samples = []
        for _ in range(n):
            await asyncio.sleep(STEP_DURATION)
            samples.append(self._raw_joints[:7].copy())
        return samples

    # ── main loop ────────────────────────────────────────────────────────

    def run(self, num_episodes: int = 1):
        executor = MultiThreadedExecutor()
        executor.add_node(self)
        spin_thread = threading.Thread(target=executor.spin, daemon=True)
        spin_thread.start()
        try:
            asyncio.run(self._run_async(num_episodes))
        finally:
            executor.shutdown()
            self._plot_debug()

    async def _run_async(self, num_episodes: int):
        loop = asyncio.get_event_loop()

        self.get_logger().info("Waiting for action servers...")
        ok = await loop.run_in_executor(
            None, lambda: self.action_executor.wait_for_servers(timeout_sec=15.0)
        )
        if not ok:
            return

        self.get_logger().info("Waiting for camera images...")
        deadline = loop.time() + 10.0
        while self.front1_image is None or self.front2_image is None or self.wrist_image is None:
            if loop.time() > deadline:
                self.get_logger().error("Cameras not ready after 10 s — check topics")
                return
            await asyncio.sleep(0.1)
        self.get_logger().info("Cameras ready.")

        self._init_episode_logs()

        for ep in range(num_episodes):
            print(f"\n{'='*50}")
            print(f"  Episode {ep + 1} / {num_episodes}")
            print(f"{'='*50}")

            step = 0
            while step < 500:
                # 1. Query policy from true current robot state
                print(f"  Querying policy at step {step}...")
                chunk = await loop.run_in_executor(None, self._fetch_chunk_sync)
                if chunk is None:
                    print("  ERROR: inference failed — stopping episode.")
                    break

                horizon = self.exec_horizon if self.exec_horizon > 0 else len(chunk)
                n = min(horizon, len(chunk), 500 - step)
                chunk = chunk[:n]
                print(f"  Executing {n} actions...")
                self.chunk_sizes_log.append(n)

                for action in chunk:
                    self.commanded_log.append(action[:7].copy())

                # 2. Execute chunk and sample actual state in parallel at STEP_DURATION rate
                exec_task = asyncio.create_task(self.action_executor.execute_chunk(chunk))
                actual_samples = await self._sample_actual(n)
                ok = await exec_task

                for s in actual_samples:
                    self.actual_log.append(s)
                self._save_logs()

                step += n
                if not ok:
                    print("  WARNING: trajectory did not complete successfully.")
                    break

            self._plot_debug()
            print(f"  Episode {ep + 1} done ({step} steps).")


def main():
    rclpy.init()
    node = OpenPIClientNode()
    num_ep = node.get_parameter("num_episodes").value
    node.run(num_episodes=num_ep)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
