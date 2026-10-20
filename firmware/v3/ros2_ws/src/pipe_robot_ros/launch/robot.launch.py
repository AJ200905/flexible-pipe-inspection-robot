"""Pi side:  ros2 launch pipe_robot_ros robot.launch.py
Options:  vision:=false   serial:=false   camera:=/path/video.mp4"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    params = os.path.join(get_package_share_directory("pipe_robot_ros"), "config", "robot_params.yaml")
    camera = LaunchConfiguration("camera")

    return LaunchDescription([
        DeclareLaunchArgument("serial", default_value="true", description="start serial_bridge"),
        DeclareLaunchArgument("vision", default_value="true", description="start vision_node"),
        DeclareLaunchArgument("camera", default_value="0", description="camera index or video file"),

        Node(package="pipe_robot_ros", executable="serial_bridge", name="serial_bridge",
             parameters=[params], output="screen", emulate_tty=True,
             condition=IfCondition(LaunchConfiguration("serial"))),

        Node(package="pipe_robot_ros", executable="vision_node", name="vision_node",
             parameters=[params, {"camera": ParameterValue(camera, value_type=str)}], output="screen", emulate_tty=True,
             condition=IfCondition(LaunchConfiguration("vision"))),
    ])
