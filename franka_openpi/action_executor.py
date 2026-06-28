import asyncio

import numpy as np
from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
from franka_msgs.action import Grasp, Move
from omni_place.Interface import MotionPlanningInterface
from rclpy.action import ActionClient
from trajectory_msgs.msg import JointTrajectoryPoint
import rclpy.logging

_log = rclpy.logging.get_logger("action_executor")

JOINT_NAMES = [
    "fer_joint1",
    "fer_joint2",
    "fer_joint3",
    "fer_joint4",
    "fer_joint5",
    "fer_joint6",
    "fer_joint7",
]

GRIPPER_CLOSE_THRESHOLD = 0.03
STEP_DURATION = 1.0 / 30  # seconds per policy step (30 Hz = saifahmad123/Teleop fps).
# Actions are delta-based internally (re-integrated by AbsoluteActions), so consecutive
# targets are spaced assuming the 30 Hz training rate. Executing slower (e.g. 20 Hz)
# stretches every motion to >1x its trained duration and drifts the re-query cadence.

# ── Gripper mode ──────────────────────────────────────────────────────────────
# "binary"     — threshold-based open/close transitions (existing behaviour)
# "continuous" — send policy gripper position directly to hardware each step
GRIPPER_MODE = "binary"


class ActionExecutor:
    def __init__(self, node):
        self._follow_jt = ActionClient(
            node, FollowJointTrajectory, "/fer_arm_controller/follow_joint_trajectory"
        )
        self._gripper_move = ActionClient(node, Move, "/fer_gripper/move")
        self._gripper_grasp = ActionClient(node, Grasp, "/fer_gripper/grasp")
        self._interface = MotionPlanningInterface(node)
        self._gripper_is_open: bool | None = None
        self._gripper_task: asyncio.Task | None = None
        self._gripper_goal_handle = None

    def wait_for_servers(self, timeout_sec: float = 15.0) -> bool:
        if not self._follow_jt.wait_for_server(timeout_sec=timeout_sec):
            _log.error("FollowJointTrajectory action server not available")
            return False
        return True

    # ── binary gripper helpers ────────────────────────────────────────────────

    async def _send_gripper(self, open_gripper: bool):
        if open_gripper:
            await self._interface.set_gripper_franka(
                width=0.08, speed=0.1, adaptive_stop=False
            )
        else:
            # Binary close: clean Move to 1 cm. adaptive_stop=False so it actually
            # reaches the target — adaptive_stop activates for width<=0.01 and cancels
            # the move on any stall, which stops it short of closing.
            await self._interface.set_gripper_franka(
                width=0.01, speed=0.05, adaptive_stop=False
            )

    async def _run_gripper_transitions(self, transitions: list[tuple[float, bool]]):
        """Execute binary open/close transitions timed from chunk start."""
        t0 = asyncio.get_event_loop().time()
        for delay, open_gripper in transitions:
            wait = delay - (asyncio.get_event_loop().time() - t0)
            if wait > 0:
                await asyncio.sleep(wait)
            _log.info(f"gripper → {'OPEN' if open_gripper else 'CLOSE'}")
            await self._send_gripper(open_gripper)

    # ── continuous gripper helpers ────────────────────────────────────────────

    async def _send_gripper_width(self, finger_pos: float):
        """Cancel the previous Move goal then send a new one — no queue buildup."""
        loop = asyncio.get_event_loop()
        if self._gripper_goal_handle is not None:
            try:
                cancel_aio: asyncio.Future = loop.create_future()
                self._gripper_goal_handle.cancel_goal_async().add_done_callback(
                    lambda f: loop.call_soon_threadsafe(
                        lambda: (
                            cancel_aio.set_result(f.result())
                            if not cancel_aio.done()
                            else None
                        )
                    )
                )
                await asyncio.wait_for(cancel_aio, timeout=0.05)
            except Exception:
                pass
            self._gripper_goal_handle = None

        width = float(np.clip(finger_pos * 2, 0.0, 0.08))
        goal = Move.Goal()
        goal.width = width
        goal.speed = 0.1
        goal_aio: asyncio.Future = loop.create_future()
        self._gripper_move.send_goal_async(goal).add_done_callback(
            lambda f: loop.call_soon_threadsafe(
                lambda: goal_aio.set_result(f.result()) if not goal_aio.done() else None
            )
        )
        try:
            self._gripper_goal_handle = await asyncio.wait_for(goal_aio, timeout=0.05)
        except asyncio.TimeoutError:
            pass

    async def _run_gripper_continuous(self, finger_positions: list[float]):
        """Send policy gripper positions step-by-step, cancelling the previous each time."""
        loop = asyncio.get_event_loop()
        t0 = loop.time()
        for i, finger_pos in enumerate(finger_positions):
            wait = i * STEP_DURATION - (loop.time() - t0)
            if wait > 0:
                await asyncio.sleep(wait)
            await self._send_gripper_width(finger_pos)

    async def cancel_gripper_async(self):
        """Cancel the active gripper goal and task. Must be called while the event loop is running."""
        if self._gripper_task and not self._gripper_task.done():
            self._gripper_task.cancel()
            try:
                await self._gripper_task
            except (asyncio.CancelledError, Exception):
                pass
        if self._gripper_goal_handle is not None:
            loop = asyncio.get_event_loop()
            try:
                cancel_aio: asyncio.Future = loop.create_future()
                self._gripper_goal_handle.cancel_goal_async().add_done_callback(
                    lambda f: loop.call_soon_threadsafe(
                        lambda: (
                            cancel_aio.set_result(f.result())
                            if not cancel_aio.done()
                            else None
                        )
                    )
                )
                await asyncio.wait_for(cancel_aio, timeout=0.5)
            except Exception:
                pass
            self._gripper_goal_handle = None

    def cancel_gripper_nowait(self):
        """Fire-and-forget cancel of the active gripper goal. Safe to call after loop closes."""
        if self._gripper_goal_handle is not None:
            try:
                self._gripper_goal_handle.cancel_goal_async()
            except Exception:
                pass
            self._gripper_goal_handle = None

    # ── main execution ────────────────────────────────────────────────────────

    async def execute_chunk(self, chunk: np.ndarray) -> bool:
        """Send the full chunk as one FollowJointTrajectory goal and await completion."""
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = list(JOINT_NAMES)

        for i, action in enumerate(chunk):
            pt = JointTrajectoryPoint()
            pt.positions = [float(p) for p in action[:7]]
            t = (i + 1) * STEP_DURATION
            pt.time_from_start = Duration(sec=int(t), nanosec=int((t % 1) * 1e9))
            goal.trajectory.points.append(pt)

        loop = asyncio.get_event_loop()

        # Bridge rclpy goal future → asyncio future
        goal_aio: asyncio.Future = loop.create_future()
        self._follow_jt.send_goal_async(goal).add_done_callback(
            lambda f: loop.call_soon_threadsafe(goal_aio.set_result, f.result())
        )
        handle = await goal_aio

        if not handle.accepted:
            _log.error("Trajectory goal rejected by controller")
            return False

        # ── gripper ───────────────────────────────────────────────────────────
        if GRIPPER_MODE == "continuous":
            finger_positions = [float(action[7]) for action in chunk]
            if self._gripper_task and not self._gripper_task.done():
                self._gripper_task.cancel()
            self._gripper_task = asyncio.create_task(
                self._run_gripper_continuous(finger_positions)
            )
        else:
            # binary: detect open/close transitions and fire at transition points
            transitions = []
            last_state = self._gripper_is_open
            for i, action in enumerate(chunk):
                gripper_open = float(action[7]) >= GRIPPER_CLOSE_THRESHOLD
                if gripper_open != last_state:
                    transitions.append((i * STEP_DURATION, gripper_open))
                    last_state = gripper_open
            if transitions:
                self._gripper_is_open = last_state
                if self._gripper_task and not self._gripper_task.done():
                    self._gripper_task.cancel()
                self._gripper_task = asyncio.create_task(
                    self._run_gripper_transitions(transitions)
                )

        # Bridge rclpy result future → asyncio future and wait
        result_aio: asyncio.Future = loop.create_future()
        handle.get_result_async().add_done_callback(
            lambda f: loop.call_soon_threadsafe(result_aio.set_result, f.result())
        )
        result = await result_aio

        ok = result.result.error_code == FollowJointTrajectory.Result.SUCCESSFUL
        if not ok:
            _log.warning(f"Trajectory ended with error code {result.result.error_code}")
        return ok
