from glob import glob

from setuptools import setup

package_name = "pipe_robot_ros"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.py")),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
        ("share/" + package_name + "/rviz", glob("rviz/*.rviz")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Anjali Ullas",
    description="Pipe inspection robot ROS 2 nodes",
    license="MIT",
    entry_points={
        "console_scripts": [
            "serial_bridge = pipe_robot_ros.serial_bridge:main",
            "vision_node = pipe_robot_ros.vision_node:main",
        ],
    },
)
