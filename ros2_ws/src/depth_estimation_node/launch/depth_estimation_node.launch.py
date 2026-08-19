"""
Standalone launch file for depth_estimation_node.

Usage on the robot (inside the Docker container, after sourcing the
workspace -- see README.md):

    ros2 launch depth_estimation_node depth_estimation_node.launch.py

To fold this into edubot_bringup's own launch files instead (so it starts
alongside nav2_launch.launch.py), see README.md "Integrating with
edubot_bringup" -- deliberately not done automatically by this package.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    config = os.path.join(
        get_package_share_directory('depth_estimation_node'),
        'config',
        'depth_estimation_params.yaml',
    )

    return LaunchDescription([
        Node(
            package='depth_estimation_node',
            executable='depth_estimation_node',
            name='depth_estimation_node',
            output='screen',
            parameters=[config],
        ),
    ])
