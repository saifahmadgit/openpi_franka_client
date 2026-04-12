import numpy as np
from geometry_msgs.msg import Point, Pose, Quaternion
from omni_place.Interface import MotionPlanningInterface
import rclpy.logging

_log = rclpy.logging.get_logger("action_executor")


class ActionExecutor:
    def __init__(self, node):
        self.interface = MotionPlanningInterface(node)

    async def execute_eef_poses(self, actions: np.ndarray) -> bool:
        """
        Execute a batch of absolute EEF poses from openpi output.
        actions: (action_horizon, 8) — [x, y, z, qw, qx, qy, qz, gripper]

        Pi0.5 outputs absolute EEF poses in world frame (z=0 at floor).
        We subtract 1.0 m from z to convert to robot base frame, then
        move to the final waypoint of the chunk.
        """
        # ── log current EEF so we can compare to policy output ───────────
        current_ps = await self.interface.planner.fr_robot_state.end_effector_pose()
        cp = current_ps.pose
        _log.info(
            f"current EEF  pos=({cp.position.x:.4f}, {cp.position.y:.4f}, {cp.position.z:.4f})"
        )

        # ── log all action steps to see the full chunk ────────────────────
        for i, a in enumerate(actions):
            _log.info(
                f"  action[{i}] pos=({a[0]:.4f}, {a[1]:.4f}, {a[2]:.4f})  "
                f"quat=({a[3]:.3f},{a[4]:.3f},{a[5]:.3f},{a[6]:.3f})  grip={a[7]:.2f}"
            )

        # ── use the final step as the absolute target ─────────────────────
        # Policy outputs poses in world frame (z=0 at floor).
        # Robot base frame origin is ~1.0 m above the floor, so subtract that.
        WORLD_TO_BASE_Z = 1.0

        goal = actions[-1]
        tx = float(goal[0])
        ty = float(goal[1])
        tz = float(goal[2]) - WORLD_TO_BASE_Z

        target = Pose()
        target.position = Point(x=tx, y=ty, z=tz)
        target.orientation = Quaternion(
            w=float(goal[3]), x=float(goal[4]), y=float(goal[5]), z=float(goal[6])
        )

        _log.info(
            f"sending target (base frame)  pos=({tx:.4f}, {ty:.4f}, {tz:.4f})  "
            f"quat=({goal[3]:.3f},{goal[4]:.3f},{goal[5]:.3f},{goal[6]:.3f})"
        )

        result = await self.interface.plan_cartesian_path(
            goal_pose=target,
            execute=True,
        )
        _log.info(f"plan_cartesian_path returned: {result}")

        gripper_cmd = float(actions[-1, 7])
        _log.info(f"gripper cmd: {gripper_cmd:.3f} → {'CLOSE' if gripper_cmd > 0.5 else 'OPEN'}")
        if gripper_cmd > 0.5:
            await self.interface.set_gripper_franka(width=0.0, speed=0.05, adaptive_stop=True)
        else:
            await self.interface.set_gripper_franka(width=0.08, speed=0.1, adaptive_stop=False)

        return result is not None
