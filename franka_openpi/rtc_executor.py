"""Streaming chunk executor for the RTC pipeline.

Difference from action_executor, and the reason this file exists:

  action_executor sends each chunk as a FollowJointTrajectory *goal* and awaits
  the result. That blocks until the chunk finishes, and the trajectory it builds
  ends at zero velocity, so the arm comes to a full stop at every chunk boundary
  even when the next chunk is already in hand.

  This one publishes to the joint_trajectory_controller's *topic* interface
  (/fer_arm_controller/joint_trajectory). Publishing is fire-and-forget, and JTC
  replaces whatever trajectory is running with the new one. Combined with issuing
  the next inference before the current chunk runs out, the tail of a chunk is
  always overwritten before the arm reaches it -- so the terminal point, and the
  deceleration into it, are never executed.

The controller is the same `fer_arm_controller` the action path drives, so the
two pipelines MUST NOT run at the same time: a publish here would yank the
trajectory out from under an active action goal.

Nothing in action_executor.py is modified; the limit constants and the gripper
path are imported from it so both pipelines stay in sync.
"""
import asyncio

import numpy as np
import rclpy.logging
from builtin_interfaces.msg import Duration
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from franka_openpi.action_executor import (
    ACC_LIMIT,
    ENFORCE_LIMITS,
    GRIPPER_CLOSE_THRESHOLD,
    JOINT_NAMES,
    VEL_LIMIT,
    ActionExecutor,
)

_log = rclpy.logging.get_logger("rtc_executor")

TRAJECTORY_TOPIC = "/fer_arm_controller/joint_trajectory"


def _waypoint_velocities(knots: np.ndarray, times: np.ndarray) -> np.ndarray:
    """Central-difference joint velocities, zero at both ends.

    Identical to action_executor's version, and it MUST stay that way: a
    trajectory whose final point carries a non-zero velocity is silently
    discarded by fer_arm_controller. Measured on hardware -- same 50 points, same
    ramp, only the terminal element differing:

        terminal |vel| = 0.000  ->  joint 7 moved +0.0706 of 0.080 commanded
        terminal |vel| = 0.008  ->  joint 7 moved +0.0004  (nothing)

    An earlier version of this file left the last waypoint moving, reasoning that
    zeroing it bakes a full stop into every chunk boundary. That reasoning was
    wrong for this pipeline: the stall is avoided by REPLACING the trajectory
    before the arm reaches the end (publish_chunk overwrites at index H-d), so
    the decelerating tail is never executed and its velocity value never matters.
    The only thing the non-zero terminal velocity achieved was to stop every
    trajectory from being accepted at all.
    """
    vel = np.zeros_like(knots)
    for k in range(1, len(knots) - 1):
        vel[k] = (knots[k + 1] - knots[k - 1]) / (times[k - 1] + times[k])
    return vel


def _time_parameterize(
    q_start: np.ndarray,
    targets: np.ndarray,
    step_duration: float,
    v_max: np.ndarray = VEL_LIMIT,
    a_max: np.ndarray = ACC_LIMIT,
    max_iter: int = 8,
):
    """Assign segment timing + waypoint velocities so joint vel/accel stay bounded.

    Same scheme as action_executor._time_parameterize (see the reasoning there),
    with two changes: step_duration is a parameter rather than a module constant,
    and the streaming _waypoint_velocities above is used.

    Returns (positions (N,7), velocities (N,7), time_from_start (N,)).
    """
    knots = np.vstack([q_start, targets])  # (N+1, 7)
    seg = np.diff(knots, axis=0)

    times = np.maximum(np.max(np.abs(seg) / v_max, axis=1), step_duration)

    for _ in range(max_iter):
        vel = _waypoint_velocities(knots, times)
        acc = (vel[1:] - vel[:-1]) / times[:, None]
        ratio = np.max(np.abs(acc) / a_max, axis=1)
        over = ratio > 1.0
        if not np.any(over):
            break
        times[over] *= np.sqrt(ratio[over])

    vel = _waypoint_velocities(knots, times)
    return knots[1:], vel[1:], np.cumsum(times)


class RTCExecutor:
    """Publishes chunks to JTC's topic interface; never blocks on completion."""

    def __init__(self, node, step_duration: float, enforce_limits: bool = ENFORCE_LIMITS):
        self._node = node
        self._step_duration = step_duration
        self._enforce_limits = enforce_limits
        # depth 1: a queued-but-superseded trajectory is never what we want on the wire.
        self._pub = node.create_publisher(JointTrajectory, TRAJECTORY_TOPIC, 1)
        # Composed purely for the gripper path (Move/Grasp action clients + the
        # binary open/close state machine), so both pipelines drive the gripper
        # identically. Its FollowJointTrajectory client is created but never used.
        self._gripper = ActionExecutor(node)

    def wait_for_servers(self, timeout_sec: float = 15.0) -> bool:
        """JTC's topic interface has no server to wait for; check subscribers instead."""
        import time

        deadline = time.monotonic() + timeout_sec
        while time.monotonic() < deadline:
            if self._pub.get_subscription_count() > 0:
                return True
            time.sleep(0.2)
        _log.error(
            f"No subscriber on {TRAJECTORY_TOPIC} after {timeout_sec:.0f}s -- is "
            f"fer_arm_controller loaded and active?"
        )
        return False

    def publish_chunk(self, chunk: np.ndarray, q_start: np.ndarray) -> np.ndarray:
        """Send `chunk` as the trajectory to execute from now, replacing any in flight.

        chunk:   (N, >=8) absolute joint targets -- 7 arm + gripper. Only [:, :8]
                 is read; the caller keeps the server's full-width array for the
                 RTC wire format and must not slice it before storing it.
        q_start: measured arm joints (7,), used only to time the first segment.

        Returns step_times (N,): seconds from publish at which each action is
        reached. The caller needs these to map elapsed wall time onto a chunk
        index, which is what prev_actions_start is derived from.
        """
        targets = np.asarray(chunk, dtype=float)[:, :7]
        if len(targets) == 0:
            return np.empty(0)

        if self._enforce_limits and q_start is not None:
            positions, velocities, times = _time_parameterize(
                np.asarray(q_start, dtype=float), targets, self._step_duration
            )
        else:
            positions = targets
            velocities = np.zeros_like(targets)
            times = np.arange(1, len(targets) + 1) * self._step_duration

        msg = JointTrajectory()
        msg.joint_names = list(JOINT_NAMES)
        # Leave header.stamp at zero: JTC then treats time_from_start as relative
        # to receipt. A wall-clock stamp would make it discard points it considers
        # already past, which at these latencies is most of the chunk.
        for i in range(len(positions)):
            pt = JointTrajectoryPoint()
            pt.positions = [float(p) for p in positions[i]]
            pt.velocities = [float(v) for v in velocities[i]]
            t = float(times[i])
            pt.time_from_start = Duration(sec=int(t), nanosec=int((t % 1) * 1e9))
            msg.points.append(pt)
        self._pub.publish(msg)

        step_times = np.asarray(times, dtype=float)
        self._schedule_gripper(chunk, step_times)
        return step_times

    def _schedule_gripper(self, chunk: np.ndarray, step_times: np.ndarray):
        """Fire binary open/close transitions timed from this publish.

        Reuses ActionExecutor's transition runner so gripper behaviour is
        identical across the two pipelines. Any transitions still pending from
        the replaced chunk are cancelled, matching the arm being replaced.
        """
        step_start = np.concatenate([[0.0], step_times[:-1]])
        transitions = []
        last_state = self._gripper._gripper_is_open
        for i, action in enumerate(chunk):
            gripper_open = float(action[7]) >= GRIPPER_CLOSE_THRESHOLD
            if gripper_open != last_state:
                transitions.append((float(step_start[i]), gripper_open))
                last_state = gripper_open
        if not transitions:
            return
        self._gripper._gripper_is_open = last_state
        task = self._gripper._gripper_task
        if task and not task.done():
            task.cancel()
        self._gripper._gripper_task = asyncio.create_task(
            self._gripper._run_gripper_transitions(transitions)
        )

    def hold(self):
        """Publish an empty trajectory: JTC cancels the active one and holds position."""
        msg = JointTrajectory()
        msg.joint_names = list(JOINT_NAMES)
        self._pub.publish(msg)

    async def cancel_gripper_async(self):
        await self._gripper.cancel_gripper_async()

    def cancel_gripper_nowait(self):
        self._gripper.cancel_gripper_nowait()

    def reset_gripper_state(self):
        """Forget the cached open/closed state so a new episode re-issues transitions."""
        self._gripper._gripper_is_open = None
