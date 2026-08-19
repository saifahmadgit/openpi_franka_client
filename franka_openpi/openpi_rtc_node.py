"""OpenPI client with real-time action chunking (RTC).

RTC -- Black, Galliker & Levine, "Real-Time Execution of Action Chunking Flow
Policies" -- removes the pause at chunk boundaries by starting the next
inference *before* the current chunk runs out, and by conditioning that
inference on the chunk already being executed so the two agree across the
inference gap.

The client half is three things:

  1. Never block on inference. The arm keeps executing the tail of the old
     chunk while the request is in flight (rtc_executor streams to the
     controller's topic interface rather than awaiting an action goal).
  2. Fire the request at chunk index H - d rather than at H.
  3. Send the previous chunk back with the request, VERBATIM, plus the index of
     the first action not yet executed.

On (3), the one trap worth restating: the config sets use_delta_joint_actions,
so the model predicts offsets from the state sent on that call and the server
adds the state back before replying. Rebasing those numbers here would pin the
arm to targets stale by exactly the motion RTC exists to bridge. So the array
that comes back from infer() is stored untouched at full (H, 14) width and sent
back untouched -- no slicing, no delta conversion. Only the execution path takes
[:, :8]. The server owns the action space; this file never reasons about it.

Separate from openpi_client_node by design: nothing here modifies the existing
pipeline. Both drive fer_arm_controller, so DO NOT RUN BOTH AT ONCE.
"""
import asyncio
import logging
import os
import time
import threading
from collections import deque
from typing import Dict, Tuple

import numpy as np
from scipy.signal import savgol_filter
import rclpy
import websockets.sync.client
from cv_bridge import CvBridge
from openpi_client import image_tools, msgpack_numpy, websocket_client_policy
from rclpy.exceptions import ParameterUninitializedException
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState

from franka_openpi.robot_compliance import RobotCompliance
from franka_openpi.rtc_executor import RTCExecutor


class _NoPingPolicy(websocket_client_policy.WebsocketClientPolicy):
    """WebsocketClientPolicy with keepalive pings disabled (see openpi_client_node)."""

    def _wait_for_server(self) -> Tuple[websockets.sync.client.ClientConnection, Dict]:
        logging.info(f"Waiting for server at {self._uri}...")
        while True:
            try:
                headers = (
                    {"Authorization": f"Api-Key {self._api_key}"} if self._api_key else None
                )
                conn = websockets.sync.client.connect(
                    self._uri,
                    compression=None,
                    max_size=None,
                    additional_headers=headers,
                    ping_interval=None,
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

STATE_DIM = 9
STATE_OUT = 9  # checkpoint norm_stats are 9-dim; sending 8 mis-aligns normalization


class OpenPIRTCNode(Node):
    def __init__(self):
        super().__init__("openpi_rtc")

        self.declare_parameter("server_host", "129.105.69.11")
        self.declare_parameter("server_port", 8000)
        self.declare_parameter("prompt", "pick up the cracker box")
        self.declare_parameter("num_episodes", 1)
        self.declare_parameter("max_steps", 500)
        self.declare_parameter("two_front_cameras", True)
        self.declare_parameter("debug_tag", "rtc")
        # Seconds per policy step. Kept at the existing pipeline's 1/5 so switching
        # to RTC does not silently change how fast the arm moves; the dataset was
        # recorded at 30 fps, so this is the knob to walk toward 1/30 separately.
        self.declare_parameter("step_duration", 0.2)
        # d, the inference latency in control steps: fire the next request at
        # index H - d, and tell the server to freeze that many actions. 0 = adapt
        # from observed round trips. See _current_d for why this is sized off the
        # tail rather than the median.
        # s, the execution horizon: how many actions of a chunk are played before
        # the next inference is fired. This is NOT H-d.
        #
        # RTC's three regions are  [0,d) hard-frozen | [d, H-s) soft-masked |
        # [H-s, H) free, so the soft mask -- the part that actually makes
        # consecutive chunks agree rather than merely touching at one point --
        # spans H-s-d actions. Firing at H-d makes the overlap exactly d, the
        # whole of it gets hard-frozen, and the soft-mask region is empty; the
        # discontinuity just moves from index 0 to index d.
        #
        # The paper uses s ~ H/2 (real-world: H=50, s_min=25). 0 = fire at H-d,
        # the degenerate behaviour, kept only for comparison runs.
        self.declare_parameter("exec_horizon", 25)
        self.declare_parameter("rtc_d", 3)
        # Extra steps added on top of the worst d seen so far. The splice index
        # must never exceed the frozen region, so this is pure safety margin.
        self.declare_parameter("rtc_d_margin", 2)
        self.declare_parameter("enforce_limits", True)
        # Optional client-side downscale before sending. OFF by default: the
        # server already resizes, which is why full-resolution frames have always
        # worked. This is a bandwidth optimisation only (2.77 MB -> 0.45 MB, most
        # of the measured round trip), NOT a correctness fix.
        #
        # It is only safe if the server resizes to this same size with
        # image_tools.resize_with_pad, in which case its own call short-circuits
        # (resize_with_pad returns early when the image is already the target
        # size) and the model input is unchanged. If the server's transform
        # differs in size or method, enabling this stacks two resamples and
        # silently shifts the input distribution. Confirm server-side before
        # setting it. Use resize_with_pad rather than a plain resize either way:
        # 640x480 -> 224x168 centred in 224x224, aspect preserved.
        self.declare_parameter("image_size", 0)
        # Escape hatch: run the streaming loop with no prev_actions on the wire, to
        # separate "streaming fixed it" from "RTC conditioning fixed it".
        self.declare_parameter("send_prev_actions", True)
        # Steps over which the residual splice discontinuity is decayed to zero.
        # 0 = off (publish the server's chunk verbatim).
        #
        # This exists because the server does NOT hard-pin the frozen region.
        # Probed directly against 129.105.69.11:8000 with prev_actions_d=18: the
        # reply carries rtc_active=True, rtc_d_eff=18, rtc_prefix_attention_horizon=38,
        # and new[0:18] still differs from prev[start:start+18] by up to 0.030 rad
        # (independent samples differ by 0.083, so the conditioning is working --
        # it is soft prefix attention over the overlap, not a freeze).
        #
        # The residual is a fixed distance in RADIANS, so it does not shrink when
        # step_duration does: at 0.15 s it is a 0.15 rad/s velocity step, at
        # 0.075 s the same 0.022 rad becomes 0.30 rad/s and four times the
        # acceleration -- once per cycle, which is what reads as lost smoothness
        # at faster cadences. Measured splice jump p50: 0.0226 rad at 0.15,
        # 0.0219 rad at 0.075. Identical, as expected.
        self.declare_parameter("splice_blend_steps", 10)
        # Savitzky-Golay window over the chunk's joint columns; 0 or 1 = off.
        #
        # The policy's chunks zigzag at the per-action level: measured over 20
        # logged chunks, 27% of consecutive per-step deltas reverse sign, with a
        # second-difference amplitude of p50 7.4 / p95 26 mrad. That is a fixed
        # amplitude in radians, so the acceleration it implies scales as
        # 1/step_duration^2 -- harmless at 0.15 s (0.33 rad/s^2), but 4.66 rad/s^2
        # p95 and 13.0 max at 0.075 s, which exceeds ACC_LIMIT (3.75-10 at
        # _LIMIT_SCALE=0.5) and is felt as vibration. The measured joints reverse
        # velocity on 11-23% of samples at that cadence.
        #
        # Evaluated on those chunks, w=7/p=2 cuts p95 zigzag accel 4.66 -> 1.02
        # rad/s^2 for <=36 mrad of worst-case path distortion (p95 9.7 mrad).
        # w=7/p=3 smooths nearly as well and distorts less (max 25 mrad) if a
        # grasp needs the shape preserved more tightly.
        self.declare_parameter("smooth_window", 7)
        self.declare_parameter("smooth_polyorder", 2)
        # ── compliance ───────────────────────────────────────────────────────
        # The arm is impedance-controlled by libfranka underneath the position
        # interface; these soften it so contact with an obstacle yields instead of
        # driving until the safety reflex trips. Empty list = leave the robot's
        # current setting alone. See robot_compliance.py.
        self.declare_parameter("joint_stiffness", [600.0, 600.0, 600.0, 600.0, 250.0, 150.0, 50.0])
        self.declare_parameter("collision_torque", [40.0, 40.0, 38.0, 36.0, 32.0, 28.0, 24.0])
        self.declare_parameter("collision_force", [40.0, 40.0, 40.0, 50.0, 50.0, 50.0])
        # Clear a reflex and continue rather than letting it end the run. The arm
        # ignores commands while in REFLEX, so without this a single collision
        # silently stalls every remaining step of the episode.
        self.declare_parameter("auto_error_recovery", True)

        host = self.get_parameter("server_host").value
        port = self.get_parameter("server_port").value
        self.prompt = self.get_parameter("prompt").value
        self.max_steps = int(self.get_parameter("max_steps").value)
        self._two_front_cameras = bool(self.get_parameter("two_front_cameras").value)
        self._step_duration = float(self.get_parameter("step_duration").value)
        self._exec_horizon = int(self.get_parameter("exec_horizon").value)
        self._rtc_d = int(self.get_parameter("rtc_d").value)
        self._d_margin = int(self.get_parameter("rtc_d_margin").value)
        self._send_prev = bool(self.get_parameter("send_prev_actions").value)
        self._splice_blend = int(self.get_parameter("splice_blend_steps").value)
        self._smooth_w = int(self.get_parameter("smooth_window").value)
        self._smooth_p = int(self.get_parameter("smooth_polyorder").value)
        self._image_size = int(self.get_parameter("image_size").value)
        tag = self.get_parameter("debug_tag").value
        self._debug_prefix = f"{tag}_" if tag else ""

        self.bridge = CvBridge()
        self.front1_image = None
        self.front2_image = None
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

        cam_mode = (
            "3-camera (front_1 + front_2 + wrist)"
            if self._two_front_cameras
            else "2-camera (front_1 + wrist)"
        )
        self.get_logger().info(f"Camera mode: {cam_mode}")
        self.get_logger().info(f"Connecting to openpi server at {host}:{port}")
        self.policy = _NoPingPolicy(host=host, port=port)
        self.get_logger().info("Connected.")

        self.executor_ = RTCExecutor(
            self,
            step_duration=self._step_duration,
            enforce_limits=bool(self.get_parameter("enforce_limits").value),
        )

        self.compliance_ = RobotCompliance(self)
        self._auto_recover = bool(self.get_parameter("auto_error_recovery").value)
        self._joint_stiffness = self._float_list_param("joint_stiffness")
        self._collision_torque = self._float_list_param("collision_torque")
        self._collision_force = self._float_list_param("collision_force")

        # ── RTC state (cleared on every episode reset) ───────────────────────
        self._prev_actions: np.ndarray | None = None  # verbatim (H, 14) server reply
        self._chunk_step_times: np.ndarray | None = None
        self._chunk_pub_t: float = 0.0
        self._chunk_offset: int = 0  # index into _prev_actions the publish started at
        self._rtc_flags_logged: bool = False

        # ── diagnostics ──────────────────────────────────────────────────────
        self._rtt_log: list[float] = []
        self._server_infer_log: list[float] = []      # server_timing.infer_ms
        self._policy_infer_log: list[float] = []      # policy_timing.infer_ms (dispatch only)
        self._commanded_log: list[dict] = []   # per publish: t, offset, step_times, actions
        self._splice_jump_log: list[np.ndarray] = []  # (7,) target jump at each splice
        self._splice_meta: list[tuple] = []
        self._d_observed: deque = deque(maxlen=20)
        self._d_floor = 0          # ratchets up on a frozen-region violation or a missed deadline
        self._d_violations = 0
        self._dropped_deadlines = 0
        self._actual_log: list[np.ndarray] = []
        self._actual_time_log: list[float] = []
        self._t0: float | None = None
        self._debug_dir = os.path.expanduser("~/Franka/src/franka_openpi/debug/rtc")
        os.makedirs(self._debug_dir, exist_ok=True)

    def _float_list_param(self, name: str) -> list[float]:
        """Read a double-array parameter, treating an empty override as 'unset'.

        `foo:="[]"` is the documented way to say "leave the robot's current value
        alone", but an empty YAML list carries no element type, so launch_ros
        cannot build a DOUBLE_ARRAY from it and passes PARAMETER_NOT_SET instead.
        get_parameter then raises rather than returning []. Both spellings mean the
        same thing to the caller, so both collapse to an empty list here.
        """
        try:
            value = self.get_parameter(name).value
        except ParameterUninitializedException:
            return []
        return [float(v) for v in (value or [])]

    # ── callbacks ────────────────────────────────────────────────────────

    def _proc_image(self, msg) -> np.ndarray:
        """ROS Image -> CHW uint8, matching openpi_client_node byte for byte.

        With image_size == 0 (the default) this is exactly what the existing
        pipeline sends: full-resolution CHW, resized server-side. The optional
        downscale runs in HWC, because resize_with_pad expects [..., H, W, C];
        the transpose to CHW happens after either way.
        """
        img = self.bridge.imgmsg_to_cv2(msg, "rgb8")  # HWC
        if self._image_size > 0:
            img = image_tools.resize_with_pad(img, self._image_size, self._image_size)
        return np.ascontiguousarray(img.transpose(2, 0, 1))  # CHW

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
        """Same camera-key mapping as openpi_client_node -- see the note there.

        cam_high <- wrist, cam_left_wrist <- front_1, cam_right_wrist <- front_2
        in 3-cam mode. Do NOT 'fix' by name; this matches how training filled them.
        """
        if self.front1_image is None or self.wrist_image is None:
            return None
        if self._two_front_cameras and self.front2_image is None:
            return None
        if self._two_front_cameras:
            images = {
                "cam_high": self.wrist_image,
                "cam_left_wrist": self.front1_image,
                "cam_right_wrist": self.front2_image,
            }
        else:
            images = {
                "cam_high": self.front1_image,
                "cam_left_wrist": self.wrist_image,
            }
        return {
            "state": self._raw_joints[:STATE_OUT].astype(np.float32),
            "images": images,
            "prompt": self.prompt,
        }

    def _fetch_chunk_sync(
        self,
        prev_actions: np.ndarray | None,
        prev_start: int | None,
        prev_d: int | None = None,
    ) -> np.ndarray | None:
        """One blocking infer(), run on a worker thread. Returns the chunk VERBATIM.

        The reply is not sliced, rebased, or converted here. Whatever width the
        server sends is what gets stored and what goes back on the next call.

        All three RTC keys travel together and are omitted together, so the server
        sees either a complete previous-chunk context or none at all.
        """
        obs = self._get_observation()
        if obs is None:
            return None
        if prev_actions is not None and prev_start is not None:
            obs["prev_actions"] = prev_actions
            obs["prev_actions_start"] = int(prev_start)
            obs["prev_actions_d"] = int(prev_d)
        try:
            result = self.policy.infer(obs)
            self._record_timing(result)
            self._record_rtc_flags(result)
            return self._smooth_chunk(np.asarray(result["actions"], dtype=np.float32))
        except Exception as e:
            self.get_logger().error(f"Infer request failed: {e}")
            return None

    def _measure_splice(self, new_chunk: np.ndarray, s_req: int, s_reply: int, d_actual: int):
        """Record the target jump at a splice, per joint and as a max.

        Indices are clamped because _chunk_index saturates at the chunk length:
        when a deadline is dropped s_reply sits exactly at H and there is no
        old[H] to compare against, so the last real action is used instead.
        """
        if self._prev_actions is None:
            return
        i_old = int(np.clip(s_reply, 0, len(self._prev_actions) - 1))
        i_new = int(np.clip(d_actual, 0, len(new_chunk) - 1))
        jump = np.asarray(new_chunk[i_new, :7], dtype=float) - np.asarray(
            self._prev_actions[i_old, :7], dtype=float
        )
        self._splice_jump_log.append(jump)
        self._splice_meta.append((s_req, s_reply, d_actual, i_old, i_new))

    def _smooth_chunk(self, chunk: np.ndarray) -> np.ndarray:
        """Savitzky-Golay the 7 joint columns of a chunk; leave everything else.

        Applied to every chunk as it arrives, priming included, so consecutive
        chunks are filtered identically -- an asymmetry here would show up as a
        step at the splice, which is the thing _blend_splice then has to absorb.

        Column 7 (gripper) is deliberately NOT filtered: it is thresholded
        against GRIPPER_CLOSE_THRESHOLD into binary open/close transitions, so
        smoothing it only moves the crossing instant. Columns 8-13 are padding.

        Runs on the inference worker thread (called from _fetch_chunk_sync), not
        the asyncio loop.
        """
        w, po = self._smooth_w, self._smooth_p
        if w <= 1 or chunk is None or len(chunk) < 3:
            return chunk
        w = min(w, len(chunk))
        if w % 2 == 0:
            w -= 1
        if w <= po:
            return chunk
        out = np.array(chunk, dtype=np.float32, copy=True)
        out[:, :7] = savgol_filter(
            np.asarray(chunk[:, :7], dtype=float), w, po, axis=0, mode="interp"
        ).astype(np.float32)
        return out

    def _blend_splice(
        self, new_chunk: np.ndarray, s_reply: int, d_actual: int
    ) -> np.ndarray:
        """Offset the new chunk so the splice is continuous, then decay the offset.

        new_chunk[d_actual] and prev[s_reply] name the same instant, so their
        difference is the step change in commanded target the arm sees at the
        splice. The server leaves ~0.02-0.03 rad of it (see splice_blend_steps).
        Add that difference back at the splice index and taper it to zero over
        the next `splice_blend_steps` actions, so the discontinuity is spread
        across ~0.75 s of motion instead of landing in a single control step.

        Shape-preserving: every action is displaced by the same decaying offset,
        so the policy's own trajectory through the chunk is untouched -- only the
        constant that made it disagree with what was already commanded is removed.
        The gripper column is left alone; it is thresholded, not interpolated.
        """
        if self._splice_blend <= 0 or self._prev_actions is None:
            return new_chunk
        i_old = int(np.clip(s_reply, 0, len(self._prev_actions) - 1))
        i_new = int(np.clip(d_actual, 0, len(new_chunk) - 1))
        delta = np.asarray(self._prev_actions[i_old, :7], dtype=float) - np.asarray(
            new_chunk[i_new, :7], dtype=float
        )
        n = min(self._splice_blend, len(new_chunk) - i_new)
        if n <= 0 or not np.any(delta):
            return new_chunk
        # Raised cosine: 1.0 at the splice (exact continuity with the last
        # commanded target) falling smoothly to 0. A linear ramp would leave a
        # slope discontinuity at both ends, which is the same problem one
        # derivative up.
        ramp = 0.5 * (1.0 + np.cos(np.pi * np.arange(n) / n))
        blended = np.array(new_chunk, dtype=np.float32, copy=True)
        blended[i_new : i_new + n, :7] += (ramp[:, None] * delta[None, :]).astype(
            np.float32
        )
        return blended

    def _record_rtc_flags(self, result: dict):
        """Log the server's own view of the RTC request, once, on the first reply.

        The reply carries rtc_active / rtc_d_eff / rtc_start / rtc_overlap /
        rtc_prefix_attention_horizon and nothing here used to read them, so a
        server silently ignoring prev_actions was indistinguishable from one
        honouring it. Probed directly, this server sets rtc_active=True and
        rtc_d_eff equal to the d that was sent, but applies prefix ATTENTION over
        the overlap rather than pinning: new[0:d] still differs from
        prev[start:start+d] by up to 0.030 rad (0.083 without conditioning). That
        is why _blend_splice exists rather than trusting the frozen region.
        """
        if self._rtc_flags_logged:
            return
        flags = {k: v for k, v in result.items() if k.startswith("rtc_")}
        if not flags:
            return
        self._rtc_flags_logged = True
        self.get_logger().info(f"Server RTC flags: {flags}")
        if not flags.get("rtc_active", False) and self._send_prev:
            self.get_logger().error(
                "Server reports rtc_active=False despite prev_actions being sent -- "
                "chunks are independent samples and every splice is a raw "
                "discontinuity. Check prev_actions_start/d against the server."
            )

    def _record_timing(self, result: dict):
        """Keep server_timing beside policy_timing.

        policy_timing.infer_ms times JAX *dispatch*, not execution -- JAX is async
        and the computation does not block until the result is converted back to
        numpy, outside that timer. server_timing.infer_ms wraps the whole
        policy.infer and is the honest number. Logging both makes the gap visible
        rather than something to rediscover later.
        """
        srv = result.get("server_timing", {}) or {}
        pol = result.get("policy_timing", {}) or {}
        if "infer_ms" in srv:
            self._server_infer_log.append(float(srv["infer_ms"]))
        if "infer_ms" in pol:
            self._policy_infer_log.append(float(pol["infer_ms"]))

    # ── chunk bookkeeping ────────────────────────────────────────────────

    def _chunk_index(self) -> int:
        """Index into _prev_actions of the first action not yet executed.

        The whole chunk is handed to the controller in one publish, so "commanded"
        in the wire-contract sense means "already played", not "already sent". The
        mapping from wall time to index goes through the step_times the executor
        computed for this publish, because velocity limiting stretches segments --
        elapsed / step_duration would drift from the truth.
        """
        if self._chunk_step_times is None or len(self._chunk_step_times) == 0:
            return 0
        elapsed = time.monotonic() - self._chunk_pub_t
        played = int(np.searchsorted(self._chunk_step_times, elapsed, side="right"))
        return self._chunk_offset + played

    def _chunk_len(self) -> int:
        return 0 if self._prev_actions is None else len(self._prev_actions)

    def _current_d(self) -> int:
        """d: how many actions the server freezes, and how early we fire.

        Sized off the TAIL, never the median. The splice index (s_reply - s_req)
        must not exceed d: past that the client resumes inside actions the server
        never froze, which reintroduces exactly the discontinuity RTC removes --
        intermittently, and only on slow calls, which is the hardest form to
        diagnose. _d_floor ratchets up whenever that is observed, so a single
        violation permanently widens the margin for the rest of the run.
        """
        h = self._chunk_len() or 50
        if self._rtc_d > 0:
            d = max(self._rtc_d, self._d_floor)
        elif self._d_observed:
            d = max(int(max(self._d_observed)) + self._d_margin, self._d_floor)
        else:
            d = max(3, self._d_floor)
        return int(np.clip(d, 1, max(1, h - 1)))

    def _fire_index(self) -> int:
        """Chunk index at which to launch the next inference -- the execution horizon s.

        Capped at H-d regardless: firing later than that leaves less than d steps
        of chunk to play while the request is in flight, which is a guaranteed
        dropped deadline. exec_horizon == 0 selects that cap directly, which is
        the degenerate no-soft-mask configuration.
        """
        h = self._chunk_len()
        d = self._current_d()
        latest = max(0, h - d)
        if self._exec_horizon <= 0:
            return latest
        return max(1, min(self._exec_horizon, latest))

    def _mask_regions(self) -> tuple[int, int, int]:
        """(hard-frozen, soft-masked, free) action counts implied by the current s and d.

        Soft-masked == 0 means RTC has nothing to smooth with: the new chunk is
        pinned for d actions and then unconstrained, so the discontinuity simply
        relocates to index d instead of index 0.
        """
        h = self._chunk_len() or 50
        s = self._fire_index()
        d = min(self._current_d(), h)
        overlap = max(0, h - s)
        hard = min(d, overlap)
        return hard, max(0, overlap - hard), h - overlap

    def _publish(self, chunk: np.ndarray, offset: int):
        """Publish chunk[offset:] and record the bookkeeping _chunk_index needs."""
        tail = chunk[offset:]
        q_start = self._raw_joints[:7].copy()
        t_pub = time.monotonic()
        step_times = self.executor_.publish_chunk(tail, q_start=q_start)
        self._prev_actions = chunk
        self._chunk_step_times = step_times
        self._chunk_pub_t = t_pub
        self._chunk_offset = offset

        # Commanded targets on a wall-clock timeline: action i of this publish is
        # reached at t_pub + step_times[i]. Recorded for every publish, including
        # the tail steps that get overwritten -- the analysis needs both what was
        # commanded and what was actually reached before replacement.
        if self._t0 is not None:
            self._commanded_log.append(
                {
                    "t": t_pub - self._t0,
                    "offset": int(offset),
                    "step_times": step_times.astype(np.float32),
                    "actions": np.asarray(chunk[offset:, :8], dtype=np.float32),
                }
            )

    def _reset_rtc_state(self):
        """Episode boundary: forget the previous chunk entirely.

        Both wire fields are cleared together -- the server reads their absence as
        'no previous chunk, sample normally', which is also what makes the first
        inference of an episode correct.
        """
        self._prev_actions = None
        self._chunk_step_times = None
        self._chunk_pub_t = 0.0
        self._chunk_offset = 0
        self.executor_.reset_gripper_state()

    # ── main loop ────────────────────────────────────────────────────────

    def run(self, num_episodes: int = 1):
        ros_exec = MultiThreadedExecutor()
        ros_exec.add_node(self)
        spin_thread = threading.Thread(target=ros_exec.spin, daemon=True)
        spin_thread.start()
        try:
            asyncio.run(self._run_async(num_episodes))
        finally:
            self.executor_.cancel_gripper_nowait()
            ros_exec.shutdown()
            self._save_logs()

    async def _run_async(self, num_episodes: int):
        loop = asyncio.get_event_loop()

        self.get_logger().info("Waiting for fer_arm_controller...")
        ok = await loop.run_in_executor(
            None, lambda: self.executor_.wait_for_servers(timeout_sec=15.0)
        )
        if not ok:
            return

        # Compliance is applied after the controller is up but before any motion,
        # and off the event loop because the service calls block. A failure here is
        # not fatal: the arm simply keeps whatever stiffness it already had.
        self.get_logger().info("Applying compliance settings...")
        await loop.run_in_executor(
            None,
            lambda: self.compliance_.apply(
                self._joint_stiffness, self._collision_torque, self._collision_force
            ),
        )

        self.get_logger().info("Waiting for camera images...")
        deadline = loop.time() + 10.0
        while (
            self.front1_image is None
            or (self._two_front_cameras and self.front2_image is None)
            or self.wrist_image is None
        ):
            if loop.time() > deadline:
                self.get_logger().error("Cameras not ready after 10 s -- check topics")
                return
            await asyncio.sleep(0.1)
        self.get_logger().info("Cameras ready.")

        sampler = asyncio.create_task(self._sample_actual())
        try:
            for ep in range(num_episodes):
                print(f"\n{'='*60}\n  RTC episode {ep + 1} / {num_episodes}\n{'='*60}")
                await self._run_episode()
                print(f"  Episode {ep + 1} done.")
                self._report()
        finally:
            sampler.cancel()
            self.executor_.hold()
            await self.executor_.cancel_gripper_async()

    async def _recover_and_reprime(self, loop) -> bool:
        """Clear a reflex, then restart chunking from where the arm actually is.

        The previous chunk cannot be resumed: the trajectory was dropped when the
        reflex hit, and the arm is wherever it stopped rather than at the index the
        step_times imply. Conditioning the next inference on that stale chunk would
        splice against actions the arm never reached, so RTC state is cleared and a
        fresh chunk is primed exactly as at episode start.
        """
        if not await loop.run_in_executor(None, self.compliance_.recover):
            print("  ERROR: could not clear reflex -- ending episode.")
            return False

        self._reset_rtc_state()
        chunk = await loop.run_in_executor(None, self._fetch_chunk_sync, None, None)
        if chunk is None:
            print("  ERROR: re-prime after reflex failed -- ending episode.")
            return False
        self._publish(chunk, offset=0)
        self.get_logger().info("Re-primed after reflex; resuming.")
        return True

    async def _run_episode(self):
        loop = asyncio.get_event_loop()
        self._reset_rtc_state()

        # First inference of the episode: nothing to condition on, so no wire
        # fields, and nothing is executing yet so blocking here is free.
        print("  Priming chunk (no prev_actions)...")
        if self._exec_horizon <= 0:
            self.get_logger().warning(
                "exec_horizon=0: firing at H-d, so the overlap equals d and the entire "
                "overlap is hard-frozen -- RTC's soft-mask region is EMPTY and the "
                "discontinuity just moves to index d. Use exec_horizon ~ H/2 (25)."
            )
        t_req = time.monotonic()
        chunk = await loop.run_in_executor(None, self._fetch_chunk_sync, None, None)
        if chunk is None:
            print("  ERROR: first inference failed -- aborting episode.")
            return
        rtt = time.monotonic() - t_req
        self._rtt_log.append(rtt)
        if self._t0 is None:
            self._t0 = t_req
        print(f"  Primed: chunk {chunk.shape} in {rtt:.2f}s (includes any JIT warm-up)")
        self._publish(chunk, offset=0)

        steps_done = 0
        cycle = 0
        while steps_done < self.max_steps:
            # A reflex leaves the arm stationary and ignoring the controller, but
            # _chunk_index is driven by wall time, not feedback -- so without this
            # check the loop keeps splicing chunks into an arm that is not moving,
            # and the whole episode plays out as a stall.
            if self._auto_recover and self.compliance_.in_reflex():
                if not await self._recover_and_reprime(loop):
                    break
                continue

            h = self._chunk_len()
            fire_at = self._fire_index()
            # Where playback of the current chunk began. Everything from here to
            # the index reached at reply time is motion the arm actually executed,
            # which is what max_steps is meant to bound.
            played_from = self._chunk_offset

            # 1. Play out the chunk until index H - d.
            while self._chunk_index() < fire_at:
                await asyncio.sleep(self._step_duration / 4)

            # 2. Fire the next inference. The arm keeps moving through the tail.
            #    d_sent is the freeze width the server will use, and must be the
            #    same d that chose fire_at -- read once so they cannot diverge.
            d_sent = self._current_d()
            s_req = self._chunk_index()
            t_req = time.monotonic()
            prev = self._prev_actions if self._send_prev else None
            start = s_req if self._send_prev else None
            task = asyncio.ensure_future(
                loop.run_in_executor(None, self._fetch_chunk_sync, prev, start, d_sent)
            )

            # 3. Wait for it without blocking the arm, and notice if we run out
            #    of chunk before the reply lands. JTC holds the final point when a
            #    trajectory expires, so a dropped deadline is a stall, not a fault.
            warned = False
            while not task.done():
                await asyncio.sleep(self._step_duration / 4)
                if not warned and self._chunk_index() >= h:
                    warned = True
                    self._dropped_deadlines += 1
                    # Widen d here too, not only on a frozen-region violation.
                    # _chunk_index saturates at h, so once the chunk runs dry
                    # d_actual clamps to d_sent and the check below can never
                    # fire -- leaving adaptive mode stuck at a d that misses
                    # every deadline. This is the only ratchet that fires when
                    # the reply lands after the chunk is already exhausted.
                    self._d_floor = d_sent + self._d_margin
                    self.get_logger().warning(
                        f"DROPPED DEADLINE: chunk exhausted at index {h} with inference "
                        f"still in flight ({time.monotonic() - t_req:.2f}s so far). Arm is "
                        f"holding the last action. Raising d to >= {self._d_floor} "
                        f"(was firing at index {fire_at} with d={d_sent})."
                    )

            new_chunk = task.result()
            rtt = time.monotonic() - t_req
            self._rtt_log.append(rtt)
            if new_chunk is None:
                print("  ERROR: inference failed -- ending episode.")
                break

            # 4. Splice. The new chunk was conditioned on prev_actions_start=s_req,
            #    so new[k] lines up with old[s_req + k]. We are now at old index
            #    s_reply, so execution resumes at new[s_reply - s_req].
            s_reply = self._chunk_index()
            d_actual = s_reply - s_req
            self._d_observed.append(d_actual)

            # The server froze new_chunk[0:d_sent]. Splicing at d_actual > d_sent
            # resumes inside actions that were never pinned to the old chunk --
            # the discontinuity RTC exists to remove, reappearing only on slow
            # calls. Ratchet d up so it cannot recur at this latency.
            if self._send_prev and d_actual > d_sent:
                self._d_violations += 1
                self._d_floor = d_actual + self._d_margin
                self.get_logger().error(
                    f"FROZEN REGION EXCEEDED: spliced at {d_actual} but only {d_sent} "
                    f"actions were frozen (rtt {rtt * 1e3:.0f} ms). Expect a jerk here. "
                    f"Raising d to >= {self._d_floor} for the rest of the run."
                )

            # Splice discontinuity: old_chunk[s_reply] and new_chunk[d_actual] name
            # the SAME instant -- the first action commanded from the new chunk is
            # the one that replaces old[s_reply]. Their difference is the step
            # change in target the arm sees at the splice, in radians.
            #
            # This is the quantity RTC removes: with the server freezing
            # new[0:d] to prev[start:start+d], new[d_actual] is pinned to
            # old[s_req + d_actual] = old[s_reply] and the jump goes to ~0. Until
            # then the two chunks are independent samples and this measures how
            # far apart they land.
            self._measure_splice(new_chunk, s_req, s_reply, d_actual)

            if d_actual >= len(new_chunk):
                self.get_logger().error(
                    f"Inference consumed the whole chunk (d={d_actual} >= H="
                    f"{len(new_chunk)}); replaying its last action only."
                )
                d_actual = len(new_chunk) - 1

            # After _measure_splice, so the diagnostics keep recording the RAW
            # jump the server hands back -- that is the number that tells you
            # whether RTC conditioning is working. The blended array is what the
            # arm executes AND what goes back on the wire next cycle: prev_actions
            # is meant to be the previously *commanded* chunk, and after blending
            # that is the blended one.
            new_chunk = self._blend_splice(new_chunk, s_reply, d_actual)
            self._publish(new_chunk, offset=int(d_actual))

            # Count what was PLAYED this cycle, not the splice offset. d_actual is
            # only the few steps consumed while inference was in flight (~3-5);
            # the arm also played everything from played_from up to the fire index,
            # so counting d_actual alone overruns max_steps by roughly 10x.
            steps_done += max(s_reply - played_from, 1)
            cycle += 1
            print(
                f"  [{cycle:3d}] played {s_reply - played_from:2d} steps "
                f"({steps_done}/{self.max_steps})  rtt {rtt * 1e3:4.0f} ms  "
                f"d={d_sent} (actual {d_actual})  resume at {d_actual}/{len(new_chunk)}"
            )

    async def _sample_actual(self):
        """Record measured joints on a fixed tick, for the post-episode plot.

        Deliberately NOT interleaved with plotting: openpi_client_node renders two
        figures inside its control loop, which measured at 1.3-1.5 s per chunk of
        dead time. Here nothing is drawn until the run ends.
        """
        while True:
            await asyncio.sleep(self._step_duration)
            if self._t0 is None:
                continue
            self._actual_log.append(self._raw_joints[:8].copy())
            self._actual_time_log.append(time.monotonic() - self._t0)

    # ── diagnostics ──────────────────────────────────────────────────────

    def _report(self):
        if not self._rtt_log:
            return
        rtt = np.array(self._rtt_log[1:]) * 1e3  # drop the JIT-warmed first call
        print("\n  " + "-" * 56)
        print("  RTC timing")
        print("  " + "-" * 56)
        if len(rtt):
            print(f"    round trip  p50 {np.percentile(rtt, 50):.0f} ms   "
                  f"p95 {np.percentile(rtt, 95):.0f} ms   max {rtt.max():.0f} ms "
                  f"({len(rtt)} calls)")
        if self._server_infer_log:
            s = np.array(self._server_infer_log)
            print(f"    server_timing.infer_ms  p50 {np.percentile(s, 50):.0f}   "
                  f"p95 {np.percentile(s, 95):.0f}   <- real model time")
        if self._policy_infer_log:
            p = np.array(self._policy_infer_log)
            print(f"    policy_timing.infer_ms  p50 {np.percentile(p, 50):.0f}   "
                  f"(JAX dispatch only -- not execution; expect it to read far low)")
        if self._d_observed:
            d = np.array(self._d_observed)
            print(f"    d observed  p50 {np.percentile(d, 50):.1f}   max {d.max()} steps"
                  f"   (d sent {self._current_d()}, firing at index {self._fire_index()} "
                  f"of {self._chunk_len()})")
        hard, soft, free = self._mask_regions()
        print(f"    RTC regions  hard-frozen {hard}  soft-masked {soft}  free {free}"
              f"   (s={self._fire_index()}, H={self._chunk_len()})")
        if soft == 0:
            print("      ^ SOFT MASK IS EMPTY -- chunks are pinned then unconstrained, so")
            print("        the discontinuity relocates rather than smooths. Lower exec_horizon.")
        if self._splice_jump_log:
            j = np.abs(np.array(self._splice_jump_log))   # (n_splices, 7)
            per_splice = j.max(axis=1)
            print(f"    splice jump  p50 {np.percentile(per_splice, 50):.4f}   "
                  f"p95 {np.percentile(per_splice, 95):.4f}   max {per_splice.max():.4f} rad"
                  f"   ({len(per_splice)} splices)")
            print(f"      per-joint mean |jump|: {np.round(j.mean(axis=0), 4).tolist()}")
            print("      ^ step change in commanded target at each chunk handover. This is")
            print("        what RTC removes -- with the server freezing new[0:d] it should")
            print("        fall toward 0. Compare this number before/after the server half.")
        print(f"    dropped deadlines: {self._dropped_deadlines}")
        if self.compliance_.reflex_count:
            print(f"    collision reflexes recovered: {self.compliance_.reflex_count}"
                  f"   <- lower joint_stiffness or raise collision_torque")
        if self._d_violations:
            print(f"    frozen-region violations: {self._d_violations}  <- each one is a "
                  f"jerk; d is now floored at {self._d_floor}")
        if self._dropped_deadlines:
            print("    ^ raise rtc_d, or shrink the observation payload; each one is a "
                  "visible stall")

    def _save_logs(self):
        if self._actual_log:
            np.save(os.path.join(self._debug_dir, self._debug_prefix + "actual.npy"),
                    np.array(self._actual_log))
            np.save(os.path.join(self._debug_dir, self._debug_prefix + "actual_times.npy"),
                    np.array(self._actual_time_log))
        if self._rtt_log:
            np.save(os.path.join(self._debug_dir, self._debug_prefix + "rtt.npy"),
                    np.array(self._rtt_log))
        if self._splice_jump_log:
            np.save(os.path.join(self._debug_dir, self._debug_prefix + "splice_jumps.npy"),
                    np.array(self._splice_jump_log))
            np.save(os.path.join(self._debug_dir, self._debug_prefix + "splice_meta.npy"),
                    np.array(self._splice_meta))
        if self._commanded_log:
            # Ragged across publishes (the tail shortens as d grows), so npz with
            # one entry per publish rather than a single stacked array.
            flat = {}
            for k, rec in enumerate(self._commanded_log):
                flat[f"t_{k}"] = np.float32(rec["t"])
                flat[f"offset_{k}"] = np.int32(rec["offset"])
                flat[f"step_times_{k}"] = rec["step_times"]
                flat[f"actions_{k}"] = rec["actions"]
            np.savez(os.path.join(self._debug_dir, self._debug_prefix + "commanded.npz"),
                     n_publishes=np.int32(len(self._commanded_log)), **flat)
        print(f"  [RTC] logs -> {self._debug_dir}")


def main():
    rclpy.init()
    node = OpenPIRTCNode()
    num_ep = node.get_parameter("num_episodes").value
    node.run(num_episodes=num_ep)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
