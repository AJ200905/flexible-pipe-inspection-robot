"""Laptop side:  ros2 launch pipe_robot_ros view.launch.py
Decompresses the camera stream locally and opens RViz with the robot view."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    rviz_cfg = os.path.join(get_package_share_directory("pipe_robot_ros"), "rviz", "pipe_robot.rviz")

    return LaunchDescription([
        # JPEG over WiFi -> raw image on the laptop only (keeps the Pi and WiFi light)
        Node(package="image_transport", executable="republish", name="annotated_decompress",
             arguments=["compressed", "raw"],
             remappings=[("in/compressed", "/camera/annotated/compressed"),
                         ("out", "/camera/annotated_view")]),

        Node(package="rviz2", executable="rviz2", name="rviz2",
             arguments=["-d", rviz_cfg], output="screen"),
    ])
