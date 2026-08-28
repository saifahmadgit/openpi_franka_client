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
    JOINT_NAMES,
    VEL_LIMIT,
    ActionExecutor,
)

_log = rclpy.logging.get_logger("rtc_executor")

TRAJECTORY_TOPIC = "/fer_arm_controller/joint_trajectory"

# Ceiling on how long the catch-up segment may be stretched, in SECONDS.
# Uncapped, a large tracking error would stall the arm for seconds while it
# creeps back onto the commanded path; the cap trades a small residual lurch on
# the worst splices for bounded progress. See _time_parameterize.
#
# Absolute, not a multiple of step_duration. It used to be 4 step_durations,
# which silently halved the catch-up budget every time step_duration was halved
# -- exactly backwards, since a faster cadence means MORE tracking error to
# absorb, not less. Measured at step_duration=0.075: 5 of 16 publishes were
# pinned at the 4-step cap (300 ms), forcing the catch-up segment above normal
# speed and putting a lurch on every republish. 0.6 s is what 4 steps meant at
# the original 0.15 s cadence, so behaviour there is unchanged.
CATCHUP_MAX_SEC = 0.6


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
    v_now: np.ndarray | None = None,
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

    # Segment 0 spans measured position -> first commanded target, so it carries
    # accumulated tracking error rather than one policy step of motion. Measured
    # on a real run it is ~4x the distance of a normal step, and giving it the
    # same step_duration as everything else makes the arm cover it at ~4x the
    # policy's own speed: a lurch at every republish followed by a long crawl,
    # which is what reads as "start and stop" at the arm. Size it by distance
    # instead, so the catch-up happens at the speed of the motion around it.
    if len(seg) > 1:
        nominal = float(np.median(np.max(np.abs(seg[1:]), axis=1)))
        if nominal > 1e-9:
            n_steps = float(np.max(np.abs(seg[0]))) / nominal
            t_catch = min(step_duration * n_steps, CATCHUP_MAX_SEC)
            # Never command the catch-up SLOWER than the arm is already moving.
            #
            # The rule above sizes segment 0 by distance, at the policy's nominal
            # speed. But the trajectory is replanned from the measured POSITION
            # while ignoring the measured VELOCITY, so whenever the arm is moving
            # faster than nominal -- which it is, because it spends every cycle
            # catching up -- the new segment 0 asks it to slow down. Measured at
            # step_duration=0.05: commanded/actual speed p50 0.89, p10 0.58, on
            # 76% of publishes. That is a brake pulse at the republish rate, and
            # an event-triggered average over the 1.4 kHz joint log shows it
            # directly: -0.118 rad/s at t=+60 ms after every publish, peak/SEM
            # 14.6, ringing out at ~4 Hz. In band terms it is 1.10 mm of EE
            # motion at 1-3 Hz against 1.99 mm of actual task motion.
            #
            # Capping t_catch at the time the gap takes at the CURRENT speed
            # removes the step. It only ever shortens segment 0, so the vel/accel
            # iteration below still bounds the result -- this cannot command
            # motion past VEL_LIMIT/ACC_LIMIT.
            if v_now is not None:
                speed = float(np.max(np.abs(np.asarray(v_now, dtype=float))))
                if speed > 1e-6:
                    t_at_speed = float(np.max(np.abs(seg[0]))) / speed
                    t_catch = min(t_catch, max(t_at_speed, step_duration))
            times[0] = max(times[0], t_catch)

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

    def publish_chunk(
        self, chunk: np.ndarray, q_start: np.ndarray, v_now: np.ndarray | None = None
    ) -> np.ndarray:
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
                np.asarray(q_start, dtype=float), targets, self._step_duration, v_now
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

        The cancel happens BEFORE the empty check, and must stay there. A chunk
        planning the whole task schedules the grasp and the eventual release
        together; once the grasp has fired, the policy is merely holding and its
        chunks contain no transitions at all. Returning early on those without
        cancelling leaves the old release pending, so it fires later and drops
        the object mid-carry -- while the arm keeps executing a plan that
        assumes it is still holding. An empty transition list is a positive
        instruction ("no change from here"), not an absence of information.
        """
        step_start = np.concatenate([[0.0], step_times[:-1]])
        transitions = self._gripper.gripper_transitions(chunk, step_start)
        task = self._gripper._gripper_task
        if task and not task.done():
            task.cancel()
        if not transitions:
            return
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
