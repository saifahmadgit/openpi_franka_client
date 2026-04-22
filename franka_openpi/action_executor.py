import numpy as np
from moveit_msgs.msg import MoveItErrorCodes
from omni_place.Interface import MotionPlanningInterface
import rclpy.logging

_log = rclpy.logging.get_logger("action_executor")

# Must match the joint names used in the MoveIt URDF / joint_states topic
JOINT_NAMES = [
    "fer_joint1",
    "fer_joint2",
    "fer_joint3",
    "fer_joint4",
    "fer_joint5",
    "fer_joint6",
    "fer_joint7",
]


class ActionExecutor:
    def __init__(self, node):
        self.interface = MotionPlanningInterface(node)

    async def execute_joint_commands(self, actions: np.ndarray) -> bool:
        """
        Execute a batch of absolute joint position commands from openpi output.
        actions: (action_horizon, 8) — [joint_0..6, gripper]

        Uses the final waypoint of the chunk as the joint target.
        """
        # log all steps so the full chunk is visible in the log
        for i, a in enumerate(actions):
            joints_str = ", ".join(f"j{j}={a[j]:.4f}" for j in range(7))
            _log.info(f"  action[{i}] {joints_str}  grip={a[7]:.3f}")

        goal = actions[-1]
        # plan_path_joint expects dict[str, float]
        goal_joint = {name: float(goal[j]) for j, name in enumerate(JOINT_NAMES)}
        gripper_cmd = float(goal[7])

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

        _log.info(f"gripper cmd: {gripper_cmd:.3f} → {'CLOSE' if gripper_cmd > 0.5 else 'OPEN'}")
        if gripper_cmd > 0.5:
            await self.interface.set_gripper_franka(width=0.0, speed=0.05, adaptive_stop=True)
        else:
            await self.interface.set_gripper_franka(width=0.08, speed=0.1, adaptive_stop=False)

        return success
