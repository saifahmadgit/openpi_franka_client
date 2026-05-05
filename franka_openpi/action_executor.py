import numpy as np
from moveit_msgs.msg import MoveItErrorCodes
from omni_place.Interface import MotionPlanningInterface
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

GRIPPER_CLOSE_THRESHOLD = 0.02  # midpoint of [0.0, 0.04]; low value = closed


class ActionExecutor:
    def __init__(self, node):
        self.interface = MotionPlanningInterface(node)

    async def execute_action(self, action: np.ndarray) -> bool:
        """
        Execute a single (8,) action: [joint_0..6, gripper].
        Joint values are absolute positions in radians; gripper in [0.0, 0.04].
        Ensembling and chunk management are handled upstream in TemporalEnsembler.
        """
        goal_joint = {name: float(action[j]) for j, name in enumerate(JOINT_NAMES)}
        gripper_cmd = float(action[7])

        _log.info(
            "sending joint target: "
            + ", ".join(f"{n}={p:.4f}" for n, p in goal_joint.items())
        )

        result = await self.interface.plan_path_joint(
            goal_joint=goal_joint,
            execute=True,
            save=False,
        )

        success = (
            result is not None
            and result.error_code.val == MoveItErrorCodes.SUCCESS
        )
        _log.info(
            f"plan_path_joint {'succeeded' if success else 'FAILED'} "
            f"(code={result.error_code.val if result is not None else 'None'})"
        )

        _log.info(f"gripper cmd: {gripper_cmd:.3f} → {'CLOSE' if gripper_cmd < GRIPPER_CLOSE_THRESHOLD else 'OPEN'}")
        if gripper_cmd < GRIPPER_CLOSE_THRESHOLD:
            await self.interface.set_gripper_franka(width=0.0, speed=0.05, adaptive_stop=True)
        else:
            await self.interface.set_gripper_franka(width=0.08, speed=0.1, adaptive_stop=False)

        return success
