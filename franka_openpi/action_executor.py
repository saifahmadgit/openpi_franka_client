import asyncio

import numpy as np
from builtin_interfaces.msg import Duration
from franka_msgs.action import Grasp, Move
from omni_place.Interface import MotionPlanningInterface
from rclpy.action import ActionClient
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
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
STEP_DURATION   = 0.1   # seconds between policy steps (controls execution rate)
REACH_DURATION  = 1.0   # seconds the controller is given to reach each waypoint
                        # higher = slower, smoother; lower = faster, more aggressive


class ActionExecutor:
    def __init__(self, node):
        self._traj_pub = node.create_publisher(
            JointTrajectory,
            "/fer_arm_controller/joint_trajectory",
            10,
        )
        self._gripper_move  = ActionClient(node, Move,  "/fer_gripper/move")
        self._gripper_grasp = ActionClient(node, Grasp, "/fer_gripper/grasp")
        self._interface = MotionPlanningInterface(node)
        self._gripper_is_open: bool | None = None

    def _publish_joints(self, positions: np.ndarray):
        """Publish a single-point trajectory. REACH_DURATION controls how
        aggressively the controller tracks each waypoint."""
        msg = JointTrajectory()
        msg.joint_names = list(JOINT_NAMES)
        pt = JointTrajectoryPoint()
        pt.positions = [float(p) for p in positions[:7]]
        pt.time_from_start = Duration(
            sec=int(REACH_DURATION),
            nanosec=int((REACH_DURATION % 1) * 1e9),
        )
        msg.points = [pt]
        self._traj_pub.publish(msg)

    async def _send_gripper(self, open_gripper: bool):
        if open_gripper:
            await self._interface.set_gripper_franka(width=0.08, speed=0.1,  adaptive_stop=False)
        else:
            await self._interface.set_gripper_franka(width=0.0,  speed=0.05, adaptive_stop=True)

    async def execute_chunk(self, chunk: np.ndarray) -> bool:
        for i, action in enumerate(chunk):
            self._publish_joints(action[:7])

            gripper_open = float(action[7]) >= GRIPPER_CLOSE_THRESHOLD
            if gripper_open != self._gripper_is_open:
                self._gripper_is_open = gripper_open
                _log.info(f"gripper → {'OPEN' if gripper_open else 'CLOSE'} at step {i}")
                asyncio.create_task(self._send_gripper(gripper_open))

            await asyncio.sleep(STEP_DURATION)
        return True

    async def execute_joint_commands(self, actions: np.ndarray) -> bool:
        action = actions[-1]
        self._publish_joints(action[:7])

        gripper_open = float(action[7]) >= GRIPPER_CLOSE_THRESHOLD
        if gripper_open != self._gripper_is_open:
            self._gripper_is_open = gripper_open
            _log.info(f"gripper → {'OPEN' if gripper_open else 'CLOSE'}")
            asyncio.create_task(self._send_gripper(gripper_open))

        await asyncio.sleep(STEP_DURATION)
        return True
