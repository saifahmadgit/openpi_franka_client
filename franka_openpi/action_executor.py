import numpy as np
from omni_place.Interface import MotionPlanningInterface
import rclpy.logging

_log = rclpy.logging.get_logger("action_executor")

JOINT_NAMES = [
    "panda_joint1",
    "panda_joint2",
    "panda_joint3",
    "panda_joint4",
    "panda_joint5",
    "panda_joint6",
    "panda_joint7",
]


class ActionExecutor:
    def __init__(self, node):
        self.interface = MotionPlanningInterface(node)

    async def execute_joint_commands(self, actions: np.ndarray) -> bool:
        """
        Execute a batch of joint position commands from openpi output.
        actions: (action_horizon, 8) — [joint_0..6, gripper]

        Uses the final waypoint of the chunk as the joint target.
        """
        # ── log all action steps to see the full chunk ────────────────────
        for i, a in enumerate(actions):
            joints_str = ", ".join(f"j{j}={a[j]:.4f}" for j in range(7))
            _log.info(f"  action[{i}] {joints_str}  grip={a[7]:.3f}")

        goal = actions[-1]
        joint_positions = [float(goal[j]) for j in range(7)]
        gripper_cmd = float(goal[7])

        _log.info(
            "sending joint target: "
            + ", ".join(f"{n}={p:.4f}" for n, p in zip(JOINT_NAMES, joint_positions))
        )

        result = await self.interface.plan_to_joint_target(
            joint_positions=joint_positions,
            execute=True,
        )
        _log.info(f"plan_to_joint_target returned: {result}")

        _log.info(f"gripper cmd: {gripper_cmd:.3f} → {'CLOSE' if gripper_cmd > 0.5 else 'OPEN'}")
        if gripper_cmd > 0.5:
            await self.interface.set_gripper_franka(width=0.0, speed=0.05, adaptive_stop=True)
        else:
            await self.interface.set_gripper_franka(width=0.08, speed=0.1, adaptive_stop=False)

        return result is not None
