from launch import LaunchDescription
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    return LaunchDescription(
        [
            # ── Front 1 RealSense (938422076779) ─────────────────────────────
            Node(
                package="realsense2_camera",
                namespace="camera/front_1",
                name="camera",
                executable="realsense2_camera_node",
                parameters=[
                    {
                        "serial_no": ParameterValue("938422076779", value_type=str),
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
            # ── Front 2 RealSense (342522070195) ─────────────────────────────
            Node(
                package="realsense2_camera",
                namespace="camera/front_2",
                name="camera",
                executable="realsense2_camera_node",
                parameters=[
                    {
                        "serial_no": ParameterValue("342522070195", value_type=str),
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
            # ── Wrist RealSense (347622076595) ────────────────────────────────
            Node(
                package="realsense2_camera",
                namespace="camera/wrist",
                name="camera",
                executable="realsense2_camera_node",
                parameters=[
                    {
                        "serial_no": ParameterValue("347622076595", value_type=str),
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
            # ── Camera viewer ─────────────────────────────────────────────────
            Node(
                package="franka_openpi",
                executable="camera_viewer",
                name="camera_viewer",
                output="screen",
            ),
        ]
    )
