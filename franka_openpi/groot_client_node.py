import asyncio
import collections
import os
import threading
import traceback

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState

from gr00t.policy.server_client import PolicyClient

from franka_openpi.action_executor import ActionExecutor, STEP_DURATION  # noqa: F401


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

ARM_DIM = 7


class GrootClientNode(Node):
    def __init__(self):
        super().__init__("groot_client")

        self.declare_parameter("server_host", "localhost")
        self.declare_parameter("server_port", 5555)
        self.declare_parameter("prompt", "pick up the object")
        self.declare_parameter("num_episodes", 5)
        self.declare_parameter("exec_horizon", 0)
        self.declare_parameter("image_h", 224)
        self.declare_parameter("image_w", 224)

        host = self.get_parameter("server_host").value
        port = self.get_parameter("server_port").value
        self.prompt = self.get_parameter("prompt").value
        self.exec_horizon = self.get_parameter("exec_horizon").value
        self._img_h = self.get_parameter("image_h").value
        self._img_w = self.get_parameter("image_w").value

        self.bridge = CvBridge()
        self._raw_joints = np.zeros(9)  # 7 arm + 2 finger

        self.get_logger().info(f"Connecting to Groot server at {host}:{port}")
        self._client = PolicyClient(host=host, port=port, timeout_ms=15000)

        modality_config = self._client.get_modality_config()
        self.get_logger().info(f"Modality config: {modality_config}")

        t_video = len(modality_config["video"].delta_indices)
        t_state = len(modality_config["state"].delta_indices)
        self.get_logger().info(f"T_video={t_video}, T_state={t_state}")

        self._t_video = t_video
        self._t_state = t_state

        # Rolling image buffers — HWC uint8 frames
        self._front1_buf = collections.deque(maxlen=t_video)
        self._front2_buf = collections.deque(maxlen=t_video)
        self._wrist_buf = collections.deque(maxlen=t_video)

        # Rolling state buffers — one entry per joint_states message
        self._arm_buf = collections.deque(maxlen=t_state)   # entries shape (7,)
        self._grip_buf = collections.deque(maxlen=t_state)  # entries shape (2,)

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

    # ── callbacks ────────────────────────────────────────────────────────

    def _proc_image(self, msg) -> np.ndarray:
        img = self.bridge.imgmsg_to_cv2(msg, "rgb8")
        if img.shape[:2] != (self._img_h, self._img_w):
            img = cv2.resize(img, (self._img_w, self._img_h), interpolation=cv2.INTER_LINEAR)
        return img  # HWC uint8

    def _front1_cb(self, msg):
        self._front1_buf.append(self._proc_image(msg))

    def _front2_cb(self, msg):
        self._front2_buf.append(self._proc_image(msg))

    def _wrist_cb(self, msg):
        self._wrist_buf.append(self._proc_image(msg))

    def _joint_cb(self, msg):
        name_to_pos = dict(zip(msg.name, msg.position))
        for i, jname in enumerate(JOINT_ORDER):
            if jname in name_to_pos:
                self._raw_joints[i] = name_to_pos[jname]
        self._arm_buf.append(self._raw_joints[:ARM_DIM].astype(np.float32))
        grip = np.clip(self._raw_joints[7:9], 0.020, 0.038).astype(np.float32)
        self._grip_buf.append(grip)

    # ── observation / inference ──────────────────────────────────────────

    def _pad_buf(self, buf: collections.deque, T: int) -> np.ndarray | None:
        """Return (T, ...) array from deque, left-padding with first frame when not yet full."""
        frames = list(buf)
        if not frames:
            return None
        while len(frames) < T:
            frames.insert(0, frames[0])
        return np.stack(frames, axis=0)

    def _get_observation(self) -> dict | None:
        if not self._front1_buf or not self._front2_buf or not self._wrist_buf or not self._arm_buf:
            return None

        front1 = self._pad_buf(self._front1_buf, self._t_video)  # (T, H, W, 3)
        front2 = self._pad_buf(self._front2_buf, self._t_video)
        wrist = self._pad_buf(self._wrist_buf, self._t_video)
        arm = self._pad_buf(self._arm_buf, self._t_state)         # (T, 7)
        grip = self._pad_buf(self._grip_buf, self._t_state)       # (T, 2)

        return {
            "video": {
                "front_1": front1[None],  # (1, T, H, W, 3)
                "front_2": front2[None],
                "wrist":   wrist[None],
            },
            "state": {
                "arm":     arm[None],     # (1, T, 7)
                "gripper": grip[None],    # (1, T, 2)
            },
            "language": {
                "task": [[self.prompt]],  # (B=1, 1)
            },
        }

    def _fetch_chunk_sync(self) -> np.ndarray | None:
        obs = self._get_observation()
        if obs is None:
            return None
        try:
            action_dict, _ = self._client.get_action(obs)
            arm = action_dict["arm"][0]       # (T_action, 7)
            grip = action_dict["gripper"][0]  # (T_action, 2)
            return np.concatenate([arm, grip], axis=-1)  # (T_action, 9)
        except Exception as e:
            self.get_logger().error(f"Infer request failed: {e}")
            return None

    # ── debug logging / plotting ─────────────────────────────────────────

    def _init_episode_logs(self):
        for fname in ("groot_commanded.npy", "groot_actual.npy", "groot_chunks.npy", "groot_debug.png"):
            p = os.path.join(self._debug_dir, fname)
            if os.path.exists(p):
                os.remove(p)
        self.commanded_log.clear()
        self.actual_log.clear()
        self.chunk_sizes_log.clear()

    def _save_logs(self):
        np.save(os.path.join(self._debug_dir, "groot_commanded.npy"), np.array(self.commanded_log))
        np.save(os.path.join(self._debug_dir, "groot_actual.npy"),    np.array(self.actual_log))
        np.save(os.path.join(self._debug_dir, "groot_chunks.npy"),    np.array(self.chunk_sizes_log))
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

            from franka_openpi.action_executor import GRIPPER_CLOSE_THRESHOLD
            fig, axes = plt.subplots(8, 1, figsize=(16, 20), sharex=True)
            fig.suptitle(f"GR00T Policy Command vs Robot State — {n} steps", fontsize=13)
            for j, ax in enumerate(axes[:7]):
                ax.scatter(steps, cmd[:, j], color="steelblue", s=10, label="commanded (policy)")
                ax.scatter(steps, act[:, j], color="tomato",    s=10, label="actual (robot state)")
                ax.set_ylabel(f"Joint {j + 1}\n(rad)", fontsize=8)
                ax.legend(loc="upper right", fontsize=7)
                ax.grid(True, alpha=0.3)
            ax_g = axes[7]
            ax_g.scatter(steps, cmd[:, 7], color="steelblue", s=10, label="commanded (policy)")
            ax_g.scatter(steps, act[:, 7], color="tomato",    s=10, label="actual (finger joint)")
            ax_g.axhline(GRIPPER_CLOSE_THRESHOLD, color="orange", linewidth=1, linestyle="--", label=f"close threshold ({GRIPPER_CLOSE_THRESHOLD})")
            ax_g.set_ylabel("Gripper\n(m)", fontsize=8)
            ax_g.legend(loc="upper right", fontsize=7)
            ax_g.grid(True, alpha=0.3)
            axes[-1].set_xlabel("step")
            plt.tight_layout()
            path = os.path.join(self._debug_dir, "groot_debug.png")
            plt.savefig(path, dpi=150)
            plt.close(fig)
            print(f"[GR00T] Debug plot saved → {path}")
        except Exception as e:
            print(f"[GR00T] Plot failed: {e}")
            traceback.print_exc()

    async def _sample_actual(self, n: int) -> list:
        """Sample actual joint state once per policy step during chunk execution."""
        samples = []
        for _ in range(n):
            await asyncio.sleep(STEP_DURATION)
            samples.append(self._raw_joints[:8].copy())
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
            self.action_executor.cancel_gripper_nowait()
            executor.shutdown()
            self._plot_debug()

    async def _run_async(self, num_episodes: int):
        try:
            await self._run_episodes(num_episodes)
        finally:
            await self.action_executor.cancel_gripper_async()

    async def _run_episodes(self, num_episodes: int):
        loop = asyncio.get_event_loop()

        self.get_logger().info("Waiting for action servers...")
        ok = await loop.run_in_executor(
            None, lambda: self.action_executor.wait_for_servers(timeout_sec=15.0)
        )
        if not ok:
            return

        self.get_logger().info("Waiting for cameras and joint states...")
        deadline = loop.time() + 10.0
        while not self._front1_buf or not self._front2_buf or not self._wrist_buf or not self._arm_buf:
            if loop.time() > deadline:
                self.get_logger().error("Sensors not ready after 10s — check topics")
                return
            await asyncio.sleep(0.1)
        self.get_logger().info("Sensors ready.")

        self._init_episode_logs()

        for ep in range(num_episodes):
            print(f"\n{'='*50}")
            print(f"  Episode {ep + 1} / {num_episodes}")
            print(f"{'='*50}")

            self._client.reset()

            step = 0
            while step < 500:
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
                    self.commanded_log.append(action[:8].copy())

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
    node = GrootClientNode()
    num_ep = node.get_parameter("num_episodes").value
    node.run(num_episodes=num_ep)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
