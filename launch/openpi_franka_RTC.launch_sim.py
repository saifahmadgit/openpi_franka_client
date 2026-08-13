"""RTC variant of openpi_franka.launch_sim.py -- sim-trained checkpoint, real arm.

Same cameras and same serial swaps as the non-RTC sim launch; the only changes
are the client node (openpi_rtc_node) and the RTC parameters.

DO NOT run this alongside openpi_franka.launch_sim.py. Both drive
fer_arm_controller, and the RTC node publishes to the controller's topic
interface, which would preempt an action goal from the other pipeline mid-motion.
"""
import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
from launch.conditions import IfCondition
from launch.substitutions import EnvironmentVariable, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

# Ensure openpi_client (installed in the project venv) is visible to the
# system-Python ROS node executable.
_VENV_SITE = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "..", "..",
    ".venv", "lib", "python3.12", "site-packages",
)
_VENV_SITE = os.path.normpath(_VENV_SITE)


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "num_episodes",
                default_value="1",
                description="Number of episodes to run",
            ),
            DeclareLaunchArgument(
                "prompt",
                default_value="pick up the red object",
                description="Task prompt for the policy",
            ),
            DeclareLaunchArgument(
                "two_front_cameras",
                default_value="true",
                description="Launch both front_1 and front_2 cameras (true) or only front_1",
            ),
            DeclareLaunchArgument(
                "exec_horizon",
                default_value="25",
                description=(
                    "s, actions played before the next inference fires. RTC's soft-mask "
                    "region spans H-s-d actions, so s must be well below H-d or the mask "
                    "is empty and chunks are pinned then unconstrained. Paper uses s~H/2 "
                    "(H=50, s_min=25). 0 = fire at H-d (degenerate, comparison only)."
                ),
            ),
            DeclareLaunchArgument(
                "rtc_d",
                default_value="3",
                description=(
                    "Inference latency in control steps; the next chunk is requested at "
                    "index H-d. 3 covers the measured p95 (510 ms) at a 0.2 s step. "
                    "0 = adapt from observed round trips."
                ),
            ),
            DeclareLaunchArgument(
                "step_duration",
                default_value="0.2",
                description=(
                    "Seconds per policy step. Matches the existing pipeline so RTC does "
                    "not silently change arm speed; the dataset is 30 fps (0.0333)."
                ),
            ),
            DeclareLaunchArgument(
                "rtc_d_margin",
                default_value="2",
                description=(
                    "Steps added on top of the worst observed d. The splice index must "
                    "never exceed the server's frozen region, so this is safety margin."
                ),
            ),
            DeclareLaunchArgument(
                "image_size",
                default_value="0",
                description=(
                    "0 = send full resolution (current behaviour; the server resizes). "
                    "224 pre-resizes with resize_with_pad to cut the payload 2.77MB->0.45MB "
                    "-- only enable once the server is confirmed to resize to the same "
                    "size with the same function, or you stack two resamples."
                ),
            ),
            DeclareLaunchArgument(
                "send_prev_actions",
                default_value="true",
                description=(
                    "false = stream chunks but send no prev_actions, to separate the "
                    "effect of non-blocking streaming from RTC conditioning"
                ),
            ),
            DeclareLaunchArgument(
                "max_steps",
                default_value="500",
                description="Policy steps before the episode is cut off",
            ),
            # Prepend venv site-packages so /usr/bin/python3 can import openpi_client.
            SetEnvironmentVariable(
                name="PYTHONPATH",
                value=[
                    _VENV_SITE + ":",
                    EnvironmentVariable("PYTHONPATH", default_value=""),
                ],
            ),
            # ── Front RealSense 1 ─────────────────────────────────────────────────
            Node(
                package="realsense2_camera",
                namespace="camera/front_1",
                name="camera",
                executable="realsense2_camera_node",
                parameters=[
                    {
                        "serial_no": ParameterValue(
                            "342522070195", value_type=str
                        ),  # SIM swap: front_1 key learned the RIGHT-side view in sim,
                        # so feed it the physically-right camera. front_1 → cam_left_wrist.
                        "rgb_camera.color_profile": "640x480x30",
                        "enable_depth": False,
                        "enable_infra1": False,
                        "enable_infra2": False,
                        "enable_gyro": False,
                        "enable_accel": False,
                    }
                ],
                output="screen",
            ),
            # ── Front RealSense 2 (skipped when two_front_cameras:=false) ─────────
            Node(
                package="realsense2_camera",
                namespace="camera/front_2",
                name="camera",
                executable="realsense2_camera_node",
                parameters=[
                    {
                        "serial_no": ParameterValue(
                            "938422076779", value_type=str
                        ),  # SIM swap: front_2 key learned the LEFT-side view in sim.
                        # front_2 → cam_right_wrist.
                        "rgb_camera.color_profile": "640x480x30",
                        "enable_depth": False,
                        "enable_infra1": False,
                        "enable_infra2": False,
                        "enable_gyro": False,
                        "enable_accel": False,
                    }
                ],
                output="screen",
                condition=IfCondition(LaunchConfiguration("two_front_cameras")),
            ),
            # ── Wrist RealSense ───────────────────────────────────────────────────
            Node(
                package="realsense2_camera",
                namespace="camera/wrist",
                name="camera",
                executable="realsense2_camera_node",
                parameters=[
                    {
                        "serial_no": ParameterValue(
                            "347622076595", value_type=str
                        ),  # dataset wrist → server key cam_high (mandatory base_0_rgb)
                        "rgb_camera.color_profile": "640x480x30",
                        "enable_depth": False,
                        "enable_infra1": False,
                        "enable_infra2": False,
                        "enable_gyro": False,
                        "enable_accel": False,
                    }
                ],
                output="screen",
            ),
            # ── Camera viewer ─────────────────────────────────────────────────────
            Node(
                package="franka_openpi",
                executable="camera_viewer",
                name="camera_viewer",
                output="screen",
            ),
            # ── OpenPI RTC client node ────────────────────────────────────────────
            Node(
                package="franka_openpi",
                executable="openpi_rtc_node",
                name="openpi_rtc",
                output="screen",
                parameters=[
                    {
                        "server_host": "129.105.69.11",
                        # 8000, not the 8001 the non-RTC launches use: that is the
                        # port serve_policy.py is actually listening on.
                        "server_port": 8000,
                        "prompt": LaunchConfiguration("prompt"),
                        "num_episodes": LaunchConfiguration("num_episodes"),
                        "two_front_cameras": LaunchConfiguration("two_front_cameras"),
                        "exec_horizon": LaunchConfiguration("exec_horizon"),
                        "rtc_d": LaunchConfiguration("rtc_d"),
                        "step_duration": LaunchConfiguration("step_duration"),
                        "send_prev_actions": LaunchConfiguration("send_prev_actions"),
                        "rtc_d_margin": LaunchConfiguration("rtc_d_margin"),
                        "image_size": LaunchConfiguration("image_size"),
                        "max_steps": LaunchConfiguration("max_steps"),
                        "debug_tag": "rtc_sim",
                    }
                ],
            ),
        ]
    )
