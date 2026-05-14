import argparse
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
    hwc = chw_uint8.transpose(1, 2, 0)  # still RGB at this point
    ok, buf = cv2.imencode(".png", hwc[:, :, ::-1])  # RGB→BGR so cv2 writes correct RGB PNG
    if not ok:
        raise RuntimeError("PNG encode failed")
    return base64.b64encode(buf.tobytes()).decode("utf-8")


class TemporalEnsembler:
    """
    Maintains a rolling buffer of (start_step, chunk) pairs.
    At each step, averages all predictions that cover it,
    weighted by recency: weight = exp(-lam * age) where age = step - start_step.
    Newer queries dominate; older ones fade out naturally.
    """

    def __init__(self, lam: float = 0.1):
        self.lam = lam
        self._buffer: list[tuple[int, np.ndarray]] = []

    def add(self, start_step: int, chunk: np.ndarray):
        self._buffer.append((start_step, chunk))

    def get(self, step: int) -> np.ndarray | None:
        preds, weights = [], []
        for start, chunk in self._buffer:
            idx = step - start
            if 0 <= idx < len(chunk):
                preds.append(chunk[idx])
                weights.append(np.exp(-self.lam * idx))
        if not preds:
            return None
        w = np.array(weights)
        w /= w.sum()
        return np.einsum("i,ij->j", w, np.array(preds))

    def prune(self, current_step: int):
        self._buffer = [
            (s, c) for s, c in self._buffer if s + len(c) > current_step
        ]

    def reset(self):
        self._buffer.clear()


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
        self.front1_stamp = 0
        self.front2_stamp = 0
        self.wrist_stamp  = 0
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
        self.chunk_sizes_log = []
        self._debug_dir = os.path.expanduser("~/Franka/src/franka_openpi/debug")
        os.makedirs(self._debug_dir, exist_ok=True)
        self.get_logger().info(f"ACT client ready → {self.infer_url}")

    # ── ROS callbacks ──────────────────────────────────────────────────────────

    def _proc_image(self, msg) -> np.ndarray:
        img = self.bridge.imgmsg_to_cv2(msg, "rgb8")
        img = cv2.resize(img, (IMG_W, IMG_H))
        return np.ascontiguousarray(img.transpose(2, 0, 1))

    def _front1_cb(self, msg):
        self.front1_image = self._proc_image(msg)
        self.front1_stamp = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec

    def _front2_cb(self, msg):
        self.front2_image = self._proc_image(msg)
        self.front2_stamp = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec

    def _wrist_cb(self, msg):
        self.wrist_image  = self._proc_image(msg)
        self.wrist_stamp  = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec

    def _joint_cb(self, msg):
        name_to_pos = dict(zip(msg.name, msg.position))
        for i, jname in enumerate(JOINT_ORDER):
            if jname in name_to_pos:
                self._raw_joints[i] = name_to_pos[jname]

    # ── Server communication ───────────────────────────────────────────────────

    def _build_payload(self) -> dict | None:
        if self.front1_image is None or self.front2_image is None or self.wrist_image is None:
            return None
        return {
            "state": self._raw_joints[:STATE_DIM].tolist(),
            "images": {
                "cam_high":        _encode_image(self.front1_image),
                "cam_left_wrist":  _encode_image(self.front2_image),
                "cam_right_wrist": _encode_image(self.wrist_image),
            },
        }

    def _fetch_chunk_sync(self) -> np.ndarray | None:
        payload = self._build_payload()
        if payload is None:
            return None
        try:
            resp = requests.post(self.infer_url, json=payload, timeout=30)
            resp.raise_for_status()
            return np.array(resp.json()["actions"])
        except Exception as e:
            self.get_logger().error(f"Infer request failed: {e}")
            return None

    def _post_reset(self):
        try:
            requests.post(self.reset_url, timeout=5)
        except Exception as e:
            self.get_logger().warn(f"Reset request failed: {e}")

    # ── Helpers ────────────────────────────────────────────────────────────────

    async def _wait_for_cameras(self):
        print("Waiting for camera images...")
        deadline = asyncio.get_event_loop().time() + 10.0
        while self.front1_image is None or self.front2_image is None or self.wrist_image is None:
            if asyncio.get_event_loop().time() > deadline:
                print("ERROR: Cameras not ready after 10 s — check topics")
                return False
            await asyncio.sleep(0.1)
        print("Cameras ready.")
        return True

    async def _init_episode_logs(self):
        for fname in ("commanded.npy", "actual.npy", "chunks.npy", "act_debug.png"):
            p = os.path.join(self._debug_dir, fname)
            if os.path.exists(p):
                os.remove(p)
        self.commanded_log.clear()
        self.actual_log.clear()
        self.chunk_sizes_log.clear()

    async def _wait_for_fresh_frames(self, after_ns: int):
        print("  Waiting for post-reset camera frames...")
        while True:
            if (self.front1_stamp > after_ns and
                    self.front2_stamp > after_ns and
                    self.wrist_stamp  > after_ns):
                break
            await asyncio.sleep(0.02)
        print("  Fresh frames received.")

    def _save_logs(self):
        np.save(os.path.join(self._debug_dir, "commanded.npy"),   np.array(self.commanded_log))
        np.save(os.path.join(self._debug_dir, "actual.npy"),      np.array(self.actual_log))
        np.save(os.path.join(self._debug_dir, "chunks.npy"),      np.array(self.chunk_sizes_log))
        if len(self.commanded_log) % 50 == 0:
            self._plot_debug()

    def _plot_debug(self):
        try:
            self._plot_debug_inner()
        except Exception as e:
            print(f"[ACT] Plot failed: {e}")

    def _plot_debug_inner(self):
        if not self.commanded_log or not self.actual_log:
            return
        cmd = np.array(self.commanded_log)
        act = np.array(self.actual_log)
        n = min(len(cmd), len(act))
        if n == 0:
            return
        cmd, act = cmd[:n], act[:n]
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

    # ── Main loop ──────────────────────────────────────────────────────────────

    def run(self, num_episodes: int = 1, temporal_ensembling: bool = False):
        executor = MultiThreadedExecutor()
        executor.add_node(self)
        spin_thread = threading.Thread(target=executor.spin, daemon=True)
        spin_thread.start()
        coro = self._run_async(num_episodes) if temporal_ensembling else self._run_simple_async(num_episodes)
        try:
            asyncio.run(coro)
        finally:
            executor.shutdown()
            self._plot_debug()

    async def _run_simple_async(self, num_episodes: int):
        """Query → execute full chunk → query again. No ensembling."""
        if await self._wait_for_cameras() is False:
            return
        await self._init_episode_logs()

        loop = asyncio.get_event_loop()

        for ep in range(num_episodes):
            print(f"\n{'='*50}")
            print(f"  Episode {ep + 1} / {num_episodes}  [simple mode]")
            print(f"{'='*50}")

            reset_time_ns = self.get_clock().now().nanoseconds
            self._post_reset()
            await self._wait_for_fresh_frames(reset_time_ns)

            step = 0
            while step < 500:
                print(f"  Querying at step {step}...")
                chunk = await loop.run_in_executor(None, self._fetch_chunk_sync)
                if chunk is None:
                    print("  ERROR: inference failed — stopping episode.")
                    break
                if len(chunk) != 100:
                    print(f"  WARNING: expected 100 actions, got {len(chunk)}")
                else:
                    print(f"  Got {len(chunk)} actions, executing...")
                self.chunk_sizes_log.append(len(chunk))

                # Clip chunk so we don't exceed 500 steps
                remaining = 500 - step
                chunk = chunk[:remaining]

                # Execute step-by-step and log actual state after each step
                for action in chunk:
                    self.commanded_log.append(action[:7].copy())
                    await self.action_executor.execute_joint_commands(action[np.newaxis])
                    self.actual_log.append(self._raw_joints[:7].copy())
                    self._save_logs()
                    step += 1

            self._plot_debug()

    async def _run_async(self, num_episodes: int):
        """Temporal ensembling: query every step, weighted-average overlapping chunks."""
        if await self._wait_for_cameras() is False:
            return
        await self._init_episode_logs()

        loop = asyncio.get_event_loop()
        ensembler = TemporalEnsembler(lam=0.1)

        for ep in range(num_episodes):
            print(f"\n{'='*50}")
            print(f"  Episode {ep + 1} / {num_episodes}")
            print(f"{'='*50}")

            ensembler.reset()
            reset_time_ns = self.get_clock().now().nanoseconds
            self._post_reset()
            await self._wait_for_fresh_frames(reset_time_ns)

            # list of (start_step, Future) — one entry per query fired
            pending_queries: list[tuple[int, asyncio.Future]] = []
            query_num = 0
            step = 0

            while step < 500:

                # ── 1. Fire a query every step (paper: query at every timestep) ──
                query_num += 1
                fut = loop.run_in_executor(None, self._fetch_chunk_sync)
                pending_queries.append((step, fut))
                print(f"  [Q{query_num}] Sent at step {step}")

                # ── 2. Collect all completed queries into ensembler ──────────────
                still_pending = []
                for start, f in pending_queries:
                    if f.done():
                        chunk = f.result()
                        if chunk is not None:
                            ensembler.add(start, chunk)
                            self.chunk_sizes_log.append(len(chunk))
                            if len(chunk) != 100:
                                print(f"\n  WARNING: expected 100 actions, got {len(chunk)} [Q@step {start}]")
                            else:
                                print(f"\n  [Q@step {start}] Received {len(chunk)} actions")
                    else:
                        still_pending.append((start, f))
                pending_queries = still_pending

                # ── 3. Get ensembled action ──────────────────────────────────────
                action = ensembler.get(step)

                if action is None:
                    # Buffer still empty — block on the oldest pending query
                    print("  Waiting for first inference...")
                    start0, fut0 = pending_queries.pop(0)
                    chunk = await fut0
                    if chunk is None:
                        print("  ERROR: first inference failed — stopping episode.")
                        break
                    ensembler.add(start0, chunk)
                    self.chunk_sizes_log.append(len(chunk))
                    if len(chunk) != 100:
                        print(f"\n  WARNING: expected 100 actions, got {len(chunk)} [Q@step {start0}]")
                    else:
                        print(f"\n  [Q@step {start0}] Received {len(chunk)} actions")
                    action = ensembler.get(step)
                    if action is None:
                        break

                # ── 4. Execute ───────────────────────────────────────────────────
                print(
                    f"    Step {step}  "
                    f"(ensembled: {len(ensembler._buffer)} chunks | pending: {len(pending_queries)})",
                    end="\r",
                )

                target = action.copy()
                delta = np.abs(action[:7] - self._raw_joints[:7])
                target[:7] = np.where(delta < 0.008, self._raw_joints[:7], action[:7])

                await self.action_executor.execute_joint_commands(target[np.newaxis])

                self.commanded_log.append(action[:7].copy())
                self.actual_log.append(self._raw_joints[:7].copy())
                self._save_logs()

                ensembler.prune(step + 1)
                step += 1

            self._plot_debug()

        # Cancel any lingering in-flight queries
        for _, f in pending_queries:
            f.cancel()


def main():
    parser = argparse.ArgumentParser(description="ACT client node")
    parser.add_argument("--temporal_ensembling", action="store_true",
                        help="Query every step and ensemble overlapping chunks (ACT paper style). "
                             "Default: query → execute full chunk → query again.")
    args, ros_args = parser.parse_known_args()

    rclpy.init(args=ros_args)
    node = ACTClientNode()
    num_ep = node.get_parameter("num_episodes").value
    node.run(num_episodes=num_ep, temporal_ensembling=args.temporal_ensembling)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
