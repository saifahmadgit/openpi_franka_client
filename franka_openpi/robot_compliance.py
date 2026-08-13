"""Impedance / collision-behaviour setup and reflex recovery for the FER arm.

The policy pipeline drives fer_arm_controller (a joint_trajectory_controller) on
the POSITION command interface. libfranka runs its own joint impedance controller
underneath that interface, so the arm is already impedance-controlled -- just very
stiffly (default 3000/3000/3000/2500/2500/2000/2000 Nm/rad). On contact with an
obstacle the controller keeps driving toward the commanded position until the
collision reflex trips, and the robot dead-stops in ROBOT_MODE_REFLEX.

Nothing here changes the control architecture. It turns two dials that already
exist on the running system:

  set_joint_stiffness           -- how hard the arm fights a position error
  set_full_collision_behavior   -- when the safety reflex trips

and adds the recovery the pipeline was missing: without an error_recovery call a
reflex is terminal, because the JTC topic interface is fire-and-forget and the
policy node never learns the arm stopped moving.

All three live on nodes spawned by franka_hardware, so they are only present when
running against real hardware -- with use_fake_hardware the state broadcaster is
not even loaded. Every entry point here degrades to a warning and a no-op rather
than blocking startup, so the same launch file still works in a mock setup.
"""
import threading

import rclpy
from franka_msgs.action import ErrorRecovery
from franka_msgs.msg import FrankaRobotState
from franka_msgs.srv import SetFullCollisionBehavior, SetJointStiffness
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup

# Node names come from franka_hardware: FrankaParamServiceServer is constructed as
# Node("service_server") and ActionServer as Node("action_server").
STIFFNESS_SRV = "/service_server/set_joint_stiffness"
COLLISION_SRV = "/service_server/set_full_collision_behavior"
RECOVERY_ACT = "/action_server/error_recovery"
ROBOT_STATE_TOPIC = "/franka_robot_state_broadcaster/robot_state"

# libfranka's defaults, kept here so the configured values can be read as a ratio
# rather than as bare magnitudes.
DEFAULT_STIFFNESS = [3000.0, 3000.0, 3000.0, 2500.0, 2500.0, 2000.0, 2000.0]


class RobotCompliance:
    """Applies compliance settings once, then watches for and clears reflexes."""

    def __init__(self, node):
        self._node = node
        self._cb = ReentrantCallbackGroup()
        self._stiffness_cli = node.create_client(
            SetJointStiffness, STIFFNESS_SRV, callback_group=self._cb
        )
        self._collision_cli = node.create_client(
            SetFullCollisionBehavior, COLLISION_SRV, callback_group=self._cb
        )
        self._recovery_cli = ActionClient(
            node, ErrorRecovery, RECOVERY_ACT, callback_group=self._cb
        )

        # robot_mode is written from the broadcaster's callback and read from the
        # asyncio loop, so it is guarded rather than assumed atomic.
        self._lock = threading.Lock()
        self._robot_mode: int | None = None
        self._reflex_count = 0
        node.create_subscription(
            FrankaRobotState, ROBOT_STATE_TOPIC, self._state_cb, 10,
            callback_group=self._cb,
        )

    # ── state ────────────────────────────────────────────────────────────────

    def _state_cb(self, msg: FrankaRobotState):
        with self._lock:
            self._robot_mode = int(msg.robot_mode)

    def in_reflex(self) -> bool:
        """True only when the broadcaster has actually reported REFLEX.

        Returns False when robot_mode is None -- i.e. the broadcaster is absent
        (mock hardware) -- so a missing state stream never looks like a fault.
        """
        with self._lock:
            return self._robot_mode == FrankaRobotState.ROBOT_MODE_REFLEX

    @property
    def reflex_count(self) -> int:
        return self._reflex_count

    # ── one-time setup ───────────────────────────────────────────────────────

    def apply(
        self,
        joint_stiffness: list[float] | None,
        collision_torque: list[float] | None,
        collision_force: list[float] | None,
        timeout_sec: float = 5.0,
    ) -> bool:
        """Push stiffness and collision thresholds. Returns False if either failed.

        These are runtime settings on the robot, not ROS parameters: they are lost
        on every franka_hardware restart, which is why this runs at node startup
        rather than being configured once out of band.
        """
        ok = True
        if joint_stiffness:
            req = SetJointStiffness.Request()
            req.joint_stiffness = [float(v) for v in joint_stiffness]
            ratio = joint_stiffness[0] / DEFAULT_STIFFNESS[0]
            ok &= self._call(
                self._stiffness_cli, req, timeout_sec,
                f"joint stiffness -> {[round(v) for v in joint_stiffness]} "
                f"(~{ratio:.0%} of libfranka default)",
            )
        if collision_torque or collision_force:
            req = SetFullCollisionBehavior.Request()
            # The service takes four torque arrays and four force arrays. Franka
            # distinguishes 'acceleration' (during ramp-up) from 'nominal' phases;
            # the pipeline has no reason to treat them differently, so one value
            # per joint fills both, upper and lower alike.
            torque = [float(v) for v in (collision_torque or [])]
            force = [float(v) for v in (collision_force or [])]
            if torque:
                req.lower_torque_thresholds_nominal = torque
                req.upper_torque_thresholds_nominal = torque
                req.lower_torque_thresholds_acceleration = torque
                req.upper_torque_thresholds_acceleration = torque
            if force:
                req.lower_force_thresholds_nominal = force
                req.upper_force_thresholds_nominal = force
                req.lower_force_thresholds_acceleration = force
                req.upper_force_thresholds_acceleration = force
            ok &= self._call(
                self._collision_cli, req, timeout_sec,
                f"collision thresholds -> torque {torque} force {force}",
            )
        return ok

    def _call(self, client, request, timeout_sec: float, what: str) -> bool:
        log = self._node.get_logger()
        if not client.wait_for_service(timeout_sec=timeout_sec):
            log.warning(
                f"{client.srv_name} unavailable -- skipping {what}. Expected when "
                f"running against mock hardware; on a real arm it means "
                f"franka_hardware is not up."
            )
            return False
        future = client.call_async(request)
        # The node is spun by a MultiThreadedExecutor on another thread, so this
        # waits on the future directly; spin_until_future_complete here would try
        # to spin an already-spinning node.
        if not _wait(future, timeout_sec):
            log.error(f"{client.srv_name} timed out setting {what}")
            return False
        result = future.result()
        if result is None or not result.success:
            err = getattr(result, "error", "no result")
            log.error(f"Failed setting {what}: {err}")
            return False
        log.info(f"Set {what}")
        return True

    # ── recovery ─────────────────────────────────────────────────────────────

    def recover(self, timeout_sec: float = 10.0) -> bool:
        """Clear a reflex via automaticErrorRecovery(). Blocking; call off-loop.

        The arm is stationary in REFLEX and refuses commands until this succeeds,
        so there is nothing to overlap with and blocking is the honest shape.
        """
        log = self._node.get_logger()
        self._reflex_count += 1
        if not self._recovery_cli.wait_for_server(timeout_sec=timeout_sec):
            log.error(f"{RECOVERY_ACT} unavailable -- cannot clear reflex")
            return False
        log.warning("Robot in REFLEX -- requesting automatic error recovery...")
        goal_future = self._recovery_cli.send_goal_async(ErrorRecovery.Goal())
        if not _wait(goal_future, timeout_sec):
            log.error("Error-recovery goal was not accepted in time")
            return False
        handle = goal_future.result()
        if handle is None or not handle.accepted:
            log.error("Error-recovery goal rejected")
            return False
        result_future = handle.get_result_async()
        if not _wait(result_future, timeout_sec):
            log.error("Error recovery did not complete in time")
            return False
        log.info("Error recovery complete -- arm should accept commands again.")
        return True


def _wait(future, timeout_sec: float) -> bool:
    """Block until `future` resolves, without spinning (the node spins elsewhere)."""
    done = threading.Event()
    future.add_done_callback(lambda _f: done.set())
    return done.wait(timeout_sec)
