import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
from launch.substitutions import EnvironmentVariable, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

# Ensure openpi_client (installed in the project venv) is visible to the
# system-Python ROS node executable.
_VENV_SITE = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "..",
    "..",
    "..",
    ".venv",
    "lib",
    "python3.12",
    "site-packages",
)
_VENV_SITE = os.path.normpath(_VENV_SITE)


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "exec_horizon",
                default_value="0",
                description="Actions to execute per chunk (0 = full chunk)",
            ),
            DeclareLaunchArgument(
                "num_episodes",
                default_value="5",
                description="Number of episodes to run",
            ),
            DeclareLaunchArgument(
                "prompt",
                default_value="pick up the orange cylinder",
                description="Task prompt for the policy",
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
            # serial_no must be ParameterValue(value_type=str) — the XML/launch
            # YAML writer converts bare numeric strings to integers, which the node
            # rejects because it already declared serial_no as a string parameter.
            Node(
                package="realsense2_camera",
                namespace="camera/front_1",
                name="camera",
                executable="realsense2_camera_node",
                parameters=[
                    {
                        "serial_no": ParameterValue(
                            "342522070195", value_type=str
                        ),  # right base camera (robot POV)
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
            # ── Front RealSense 2 ─────────────────────────────────────────────────
            Node(
                package="realsense2_camera",
                namespace="camera/front_2",
                name="camera",
                executable="realsense2_camera_node",
                parameters=[
                    {
                        "serial_no": ParameterValue(
                            "233522075872", value_type=str
                        ),  # left base camera (robot POV)
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
                        ),  # wrist camera
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
            # ── OpenPI client node ────────────────────────────────────────────────
            Node(
                package="franka_openpi",
                executable="openpi_client_node",
                name="openpi_client",
                parameters=[
                    {
                        "server_host": "129.105.69.11",
                        "server_port": 8000,
                        "prompt": LaunchConfiguration("prompt"),
                        "num_episodes": LaunchConfiguration("num_episodes"),
                        "exec_horizon": LaunchConfiguration("exec_horizon"),
                    }
                ],
            ),
        ]
    )
