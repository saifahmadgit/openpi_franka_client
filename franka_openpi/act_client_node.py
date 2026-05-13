import asyncio
import base64
import os
import threading

import cv2
import matplotlib
matplotlib.use("Agg")
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
        self.commanded_log = []
        self.actual_log = []
        self._debug_dir = os.path.expanduser("~/Franka/src/franka_openpi/debug")
        os.makedirs(self._debug_dir, exist_ok=True)
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

    def _plot_debug(self):
        try:
            self._plot_debug_inner()
        except Exception as e:
            print(f"[ACT] Plot failed: {e}")

    def _plot_debug_inner(self):
        if not self.commanded_log or not self.actual_log:
            return
        cmd = np.array(self.commanded_log)   # (N, 7)
        act = np.array(self.actual_log)       # (N, 7)
        n = min(len(cmd), len(act))
        if n == 0:
            return
        cmd = cmd[:n]
        act = act[:n]
        steps = np.arange(n)

        joint_labels = [f"Joint {i + 1}" for i in range(7)]
        fig, axes = plt.subplots(7, 1, figsize=(16, 18), sharex=True)
        fig.suptitle(f"Policy Command vs Robot State — {n} steps", fontsize=13)

        for j, ax in enumerate(axes):
            ax.scatter(steps, cmd[:, j], color="steelblue", s=10, label="commanded (policy)")
            ax.scatter(steps, act[:, j], color="tomato",    s=10, label="actual (robot state)")
            ax.set_ylabel(joint_labels[j] + "\n(rad)", fontsize=8)
            ax.legend(loc="upper right", fontsize=7)
            ax.grid(True, alpha=0.3)

        axes[-1].set_xlabel("step")
        plt.tight_layout()
        path = os.path.join(self._debug_dir, "act_debug.png")
        plt.savefig(path, dpi=150)
        plt.close(fig)
        print(f"[ACT] Debug plot saved → {path}")

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
        print("Waiting for camera images...")
        deadline = asyncio.get_event_loop().time() + 10.0
        while self.front1_image is None or self.front2_image is None or self.wrist_image is None:
            if asyncio.get_event_loop().time() > deadline:
                print("ERROR: Cameras not ready after 10 s — check topics")
                return
            await asyncio.sleep(0.1)
        print("Cameras ready.")

        loop = asyncio.get_event_loop()

        # Clear any results from a previous run
        import os
        for fname in ("commanded.npy", "actual.npy", "act_debug.png"):
            p = os.path.join(self._debug_dir, fname)
            if os.path.exists(p):
                os.remove(p)
        self.commanded_log.clear()
        self.actual_log.clear()

        for ep in range(num_episodes):
            print(f"\n{'='*50}")
            print(f"  Episode {ep + 1} / {num_episodes}")
            print(f"{'='*50}")
            self._post_reset()

            total_steps = 0
            chunk_num = 0
            while total_steps < 500:
                # 1. Query
                chunk_num += 1
                print(f"\n  [Chunk {chunk_num}] Querying server...")
                chunk = await loop.run_in_executor(None, lambda: self._fetch_chunk(loop))
                if chunk is None:
                    print(f"  [Chunk {chunk_num}] ERROR: fetch failed — stopping episode.")
                    break
                print(f"  [Chunk {chunk_num}] Received {len(chunk)} actions")

                # 2. Execute all actions in chunk, record per step
                for i, action in enumerate(chunk):
                    print(f"    Executing {i + 1} of {len(chunk)}  (chunk {chunk_num})", end="\r")
                    await self.action_executor.execute_joint_commands(action[np.newaxis])
                    self.commanded_log.append(action[:7].copy())
                    self.actual_log.append(self._raw_joints[:7].copy())
                    total_steps += 1
                    np.save(os.path.join(self._debug_dir, "commanded.npy"), np.array(self.commanded_log))
                    np.save(os.path.join(self._debug_dir, "actual.npy"),    np.array(self.actual_log))
                    if total_steps >= 500:
                        break

                print(f"    Executing {min(i + 1, len(chunk))} of {len(chunk)}  (chunk {chunk_num}) — done")
                self._plot_debug()
                # 3. Loop back → query again


def main():
    rclpy.init()
    node = ACTClientNode()
    num_ep = node.get_parameter("num_episodes").value
    node.run(num_episodes=num_ep)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
