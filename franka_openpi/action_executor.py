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

GRIPPER_CLOSE_THRESHOLD = 0.02
STEP_DURATION = 0.1  # seconds per policy step — sets trajectory timing


class ActionExecutor:
    def __init__(self, node):
        self._follow_jt = ActionClient(
            node, FollowJointTrajectory, "/fer_arm_controller/follow_joint_trajectory"
        )
        self._gripper_move = ActionClient(node, Move, "/fer_gripper/move")
        self._gripper_grasp = ActionClient(node, Grasp, "/fer_gripper/grasp")
        self._interface = MotionPlanningInterface(node)
        self._gripper_is_open: bool | None = None

    def wait_for_servers(self, timeout_sec: float = 15.0) -> bool:
        if not self._follow_jt.wait_for_server(timeout_sec=timeout_sec):
            _log.error("FollowJointTrajectory action server not available")
            return False
        return True

    async def _send_gripper(self, open_gripper: bool):
        if open_gripper:
            await self._interface.set_gripper_franka(
                width=0.08, speed=0.1, adaptive_stop=False
            )
        else:
            await self._interface.set_gripper_franka(
                width=0.0, speed=0.05, adaptive_stop=True
            )

    async def _send_gripper_delayed(self, open_gripper: bool, delay: float):
        await asyncio.sleep(delay)
        await self._send_gripper(open_gripper)

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

        # Schedule gripper transitions timed to their position within the chunk
        for i, action in enumerate(chunk):
            gripper_open = float(action[7]) >= GRIPPER_CLOSE_THRESHOLD
            if gripper_open != self._gripper_is_open:
                self._gripper_is_open = gripper_open
                _log.info(f"gripper → {'OPEN' if gripper_open else 'CLOSE'} at step {i}")
                asyncio.create_task(
                    self._send_gripper_delayed(gripper_open, i * STEP_DURATION)
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
