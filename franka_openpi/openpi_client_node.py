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

from franka_openpi.action_executor import ActionExecutor, STEP_DURATION, smooth_chunk


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
# Policy expects the full 9-dim state [j1..j7, finger_joint1, finger_joint2].
# The checkpoint's norm_stats are 9-dim; sending 8 mis-aligns normalization
# (the gripper value would be normalized against arm-joint-7's stats).
STATE_OUT = 9


class OpenPIClientNode(Node):
    def __init__(self):
        super().__init__("openpi_client")

        self.declare_parameter("server_host", "192.168.1.X")
        self.declare_parameter("server_port", 8000)
        self.declare_parameter("prompt", "pick up the orange cylinder")
        self.declare_parameter("num_episodes", 5)
        self.declare_parameter("exec_horizon", 0)  # 0 = execute full chunk
        self.declare_parameter("two_front_cameras", True)
        self.declare_parameter("debug_tag", "")  # filename prefix for debug outputs (e.g. "sim")

        host = self.get_parameter("server_host").value
        port = self.get_parameter("server_port").value
        self.prompt = self.get_parameter("prompt").value
        self.exec_horizon = self.get_parameter("exec_horizon").value
        self._two_front_cameras: bool = bool(self.get_parameter("two_front_cameras").value)
        # Prefix so sim and real runs write to distinct files and a stale plot
        # from one can't masquerade as the other's output.
        tag = self.get_parameter("debug_tag").value
        self._debug_prefix = f"{tag}_" if tag else ""

        self.bridge = CvBridge()
        self.front1_image = None
        self.front2_image = None  # stays None (unused) in 1-front-camera mode
        self.wrist_image = None
        self._raw_joints = np.zeros(STATE_DIM)

        self.create_subscription(
            Image, "/camera/front_1/camera/color/image_raw", self._front1_cb, 10
        )
        if self._two_front_cameras:
            self.create_subscription(
                Image, "/camera/front_2/camera/color/image_raw", self._front2_cb, 10
            )
        self.create_subscription(
            Image, "/camera/wrist/camera/color/image_raw", self._wrist_cb, 10
        )
        self.create_subscription(JointState, "/joint_states", self._joint_cb, 10)

        cam_mode = "3-camera (front_1 + front_2 + wrist)" if self._two_front_cameras else "2-camera (front_1 + wrist)"
        self.get_logger().info(f"Camera mode: {cam_mode}")

        self.get_logger().info(f"Connecting to openpi server at {host}:{port}")
        self.policy = _NoPingPolicy(host=host, port=port)
        self.get_logger().info("Connected.")

        self.action_executor = ActionExecutor(self)
        self.commanded_log = []
        self.smoothed_log = []      # what was actually sent to the controller
        self.actual_log = []
        self.actual_time_log = []   # wall-clock time (s, rel. to first request) per actual sample
        self.query_spans_log = []   # (t_request, t_received) per chunk — the inference wait
        self.chunk_sizes_log = []
        self._t0 = None             # monotonic origin: set at the very first chunk request
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
        if self.front1_image is None or self.wrist_image is None:
            return None
        if self._two_front_cameras and self.front2_image is None:
            return None
        if self._two_front_cameras:
            # 3-cam server key ← dataset source mapping (how training filled the keys):
            #   cam_high        ← wrist    (mandatory base_0_rgb)
            #   cam_left_wrist  ← front_1
            #   cam_right_wrist ← front_2
            # repack_transforms run only at training time, so the client must
            # populate these keys itself to match. Do NOT "fix" by name.
            images: dict = {
                "cam_high":        self.wrist_image,
                "cam_left_wrist":  self.front1_image,
                "cam_right_wrist": self.front2_image,
            }
        else:
            # 2-cam: front_1→cam_high, wrist→cam_left_wrist
            # Training used cam_left_wrist for the second camera; cam_right_wrist was always
            # black/masked. Sending wrist as cam_right_wrist would feed it into the wrong slot.
            images = {
                "cam_high":       self.front1_image,
                "cam_left_wrist": self.wrist_image,
            }
        return {
            "state": self._raw_joints[:STATE_OUT].astype(np.float32),
            "images": images,
            "prompt": self.prompt,
        }

    def _fetch_chunk_sync(self) -> np.ndarray | None:
        obs = self._get_observation()
        if obs is None:
            return None
        try:
            result = self.policy.infer(obs)
            # Server returns (16, 14); columns 8-13 are padding from the bimanual Aloha
            # default. Only the first 8 are meaningful: 7 arm joints + 1 gripper.
            return np.array(result["actions"])[:, :8]
        except Exception as e:
            self.get_logger().error(f"Infer request failed: {e}")
            return None

    # ── debug logging / plotting ─────────────────────────────────────────

    def _init_episode_logs(self):
        for fname in (
            "commanded.npy", "smoothed.npy", "actual.npy", "chunks.npy", "openpi_debug.png",
            "actual_times.npy", "query_spans.npy", "openpi_debug_time.png",
        ):
            p = os.path.join(self._debug_dir, self._debug_prefix + fname)
            if os.path.exists(p):
                os.remove(p)
        self.commanded_log.clear()
        self.smoothed_log.clear()
        self.actual_log.clear()
        self.actual_time_log.clear()
        self.query_spans_log.clear()
        self.chunk_sizes_log.clear()
        self._t0 = None

    def _save_logs(self):
        np.save(os.path.join(self._debug_dir, self._debug_prefix + "commanded.npy"), np.array(self.commanded_log))
        np.save(os.path.join(self._debug_dir, self._debug_prefix + "smoothed.npy"),  np.array(self.smoothed_log))
        np.save(os.path.join(self._debug_dir, self._debug_prefix + "actual.npy"),    np.array(self.actual_log))
        np.save(os.path.join(self._debug_dir, self._debug_prefix + "chunks.npy"),    np.array(self.chunk_sizes_log))
        np.save(os.path.join(self._debug_dir, self._debug_prefix + "actual_times.npy"), np.array(self.actual_time_log))
        np.save(os.path.join(self._debug_dir, self._debug_prefix + "query_spans.npy"),  np.array(self.query_spans_log))
        # Refresh the plot after every chunk. The old `% 50` gate assumed 50-step
        # chunks and never fired for small exec_horizon values (e.g. 10 → 10,20,30
        # never hits a multiple of 50), so no plot appeared until an episode ended.
        self._plot_debug()
        self._plot_debug_time()

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
            # smoothed is logged alongside commanded, so it is at least as long as cmd;
            # guard anyway so a partially-written log can still be plotted.
            sm = np.array(self.smoothed_log)[:n] if len(self.smoothed_log) >= n else None
            steps = np.arange(n)

            from franka_openpi.action_executor import GRIPPER_CLOSE_THRESHOLD, SMOOTH_ENABLE
            fig, axes = plt.subplots(8, 1, figsize=(20, 32), sharex=True)
            smooth_note = " (smoothing off)" if not SMOOTH_ENABLE else ""
            fig.suptitle(f"Policy Command vs Smoothed vs Robot State — {n} steps{smooth_note}", fontsize=22)
            for j, ax in enumerate(axes[:7]):
                ax.scatter(steps, cmd[:, j], color="steelblue", s=18, label="commanded (policy)")
                if sm is not None:
                    ax.scatter(steps, sm[:, j], color="seagreen", s=14, marker="x", label="smoothed (sent to controller)")
                ax.scatter(steps, act[:, j], color="tomato",    s=18, label="actual (robot state)")
                ax.set_ylabel(f"Joint {j + 1}\n(rad)", fontsize=16)
                ax.legend(loc="upper right", fontsize=14)
                ax.tick_params(axis="both", labelsize=13)
                ax.grid(True, alpha=0.3)
            ax_g = axes[7]
            ax_g.scatter(steps, cmd[:, 7], color="steelblue", s=18, label="commanded (policy)")
            ax_g.scatter(steps, act[:, 7], color="tomato",    s=18, label="actual (finger joint)")
            ax_g.axhline(GRIPPER_CLOSE_THRESHOLD, color="orange", linewidth=1.5, linestyle="--", label=f"close threshold ({GRIPPER_CLOSE_THRESHOLD})")
            ax_g.set_ylabel("Gripper\n(m)", fontsize=16)
            ax_g.legend(loc="upper right", fontsize=14)
            ax_g.tick_params(axis="both", labelsize=13)
            ax_g.grid(True, alpha=0.3)
            axes[-1].set_xlabel("step", fontsize=16)
            plt.tight_layout()
            path = os.path.join(self._debug_dir, self._debug_prefix + "openpi_debug.png")
            plt.savefig(path, dpi=150)
            plt.close(fig)
            print(f"[OpenPI] Debug plot saved → {path}")
        except Exception as e:
            print(f"[OpenPI] Plot failed: {e}")

    def _plot_debug_time(self):
        """Actual robot state vs wall-clock time — the SAME actual dots as _plot_debug,
        placed on a real-time x-axis (t=0 at the first chunk request) so the idle time
        between reaching a chunk's end and receiving the next chunk is visible as a gap
        + shaded band. No commanded/server points here — only the actual trajectory."""
        try:
            if not self.actual_log or not self.actual_time_log:
                return
            act = np.array(self.actual_log)
            t = np.array(self.actual_time_log)
            n = min(len(act), len(t))
            if n == 0:
                return
            act, t = act[:n], t[:n]
            waits = np.array(self.query_spans_log) if self.query_spans_log else np.empty((0, 2))
            total_wait = float(np.sum(waits[:, 1] - waits[:, 0])) if len(waits) else 0.0

            from franka_openpi.action_executor import GRIPPER_CLOSE_THRESHOLD
            fig, axes = plt.subplots(8, 1, figsize=(20, 32), sharex=True)
            fig.suptitle(
                f"Actual Robot State vs Time — {n} samples, {len(waits)} chunk requests, "
                f"{total_wait:.1f}s total inference wait",
                fontsize=22,
            )

            def _shade_waits(ax):
                for k, (ws, we) in enumerate(waits):
                    ax.axvspan(ws, we, color="gold", alpha=0.25,
                               label="waiting for chunk" if k == 0 else None)

            for j, ax in enumerate(axes[:7]):
                _shade_waits(ax)
                ax.scatter(t, act[:, j], color="tomato", s=18, label="actual (robot state)")
                ax.set_ylabel(f"Joint {j + 1}\n(rad)", fontsize=16)
                ax.legend(loc="upper right", fontsize=14)
                ax.tick_params(axis="both", labelsize=13)
                ax.grid(True, alpha=0.3)
            ax_g = axes[7]
            _shade_waits(ax_g)
            ax_g.scatter(t, act[:, 7], color="tomato", s=18, label="actual (finger joint)")
            ax_g.axhline(GRIPPER_CLOSE_THRESHOLD, color="orange", linewidth=1.5, linestyle="--",
                         label=f"close threshold ({GRIPPER_CLOSE_THRESHOLD})")
            ax_g.set_ylabel("Gripper\n(m)", fontsize=16)
            ax_g.legend(loc="upper right", fontsize=14)
            ax_g.tick_params(axis="both", labelsize=13)
            ax_g.grid(True, alpha=0.3)
            axes[-1].set_xlabel("time (s)", fontsize=16)
            plt.tight_layout()
            path = os.path.join(self._debug_dir, self._debug_prefix + "openpi_debug_time.png")
            plt.savefig(path, dpi=150)
            plt.close(fig)
            print(f"[OpenPI] Time plot saved → {path}")
        except Exception as e:
            print(f"[OpenPI] Time plot failed: {e}")

    async def _sample_actual(self, n: int) -> tuple[list, list]:
        """Sample actual joint state once per policy step during chunk execution.

        Also returns the wall-clock time of each sample (s, relative to the run's first
        chunk request) so the trajectory can be plotted against real time.
        """
        samples, times = [], []
        for _ in range(n):
            await asyncio.sleep(STEP_DURATION)
            samples.append(self._raw_joints[:8].copy())
            times.append(time.monotonic() - self._t0)
        return samples, times

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
            self._plot_debug_time()

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
        while (
            self.front1_image is None
            or (self._two_front_cameras and self.front2_image is None)
            or self.wrist_image is None
        ):
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
                # 1. Query policy from true current robot state.
                #    Time the request→receive span: this is the idle window the robot
                #    spends waiting after reaching the previous chunk's last point.
                print(f"  Querying policy at step {step}...")
                t_req = time.monotonic()
                if self._t0 is None:
                    self._t0 = t_req  # timeline origin = first chunk request
                chunk = await loop.run_in_executor(None, self._fetch_chunk_sync)
                t_recv = time.monotonic()
                if chunk is None:
                    print("  ERROR: inference failed — stopping episode.")
                    break
                self.query_spans_log.append((t_req - self._t0, t_recv - self._t0))

                horizon = self.exec_horizon if self.exec_horizon > 0 else len(chunk)
                n = min(horizon, len(chunk), 500 - step)
                chunk = chunk[:n]
                print(f"  Executing {n} actions...")
                self.chunk_sizes_log.append(n)

                for action in chunk:
                    self.commanded_log.append(action[:8].copy())

                # 2. Smooth the intermediate waypoints before execution. First and last
                #    waypoints are reproduced exactly, so the arm still hits the poses the
                #    policy asked for at the chunk boundaries — only the step-to-step noise
                #    in between (the source of the accel spikes / table shake) is removed.
                #    The gripper column is untouched, so the transition timing computed
                #    inside execute_chunk is unaffected.
                smoothed = smooth_chunk(chunk)
                for action in smoothed:
                    self.smoothed_log.append(action[:8].copy())

                # 3. Execute chunk and sample actual state in parallel at STEP_DURATION rate.
                #    Pass the current measured arm joints so the executor can time-parameterize
                #    the chunk (incl. the jump to action[0]) within velocity/accel limits.
                q_start = self._raw_joints[:7].copy()
                exec_task = asyncio.create_task(
                    self.action_executor.execute_chunk(smoothed, q_start=q_start)
                )
                actual_samples, actual_times = await self._sample_actual(n)
                ok = await exec_task

                for s in actual_samples:
                    self.actual_log.append(s)
                self.actual_time_log.extend(actual_times)
                self._save_logs()

                step += n
                if not ok:
                    print("  WARNING: trajectory did not complete successfully.")
                    break

            self._plot_debug()
            self._plot_debug_time()
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
