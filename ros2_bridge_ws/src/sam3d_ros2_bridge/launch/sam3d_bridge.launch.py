#!/usr/bin/env python3
"""
Launch file for SAM 3D Objects ROS2 Bridge

This launch file starts the ROS2 action server that connects to the ZeroMQ inference server.

Usage:
    ros2 launch sam3d_ros2_bridge sam3d_bridge.launch.py

    Or with custom parameters:
    ros2 launch sam3d_ros2_bridge sam3d_bridge.launch.py zmq_host:=192.168.1.100 zmq_port:=5556
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # Declare launch arguments
    zmq_host_arg = DeclareLaunchArgument(
        'zmq_host',
        default_value='localhost',
        description='Hostname or IP address of the ZeroMQ inference server'
    )

    zmq_port_arg = DeclareLaunchArgument(
        'zmq_port',
        default_value='5555',
        description='Port number of the ZeroMQ inference server'
    )

    # Create the action server node
    action_server_node = Node(
        package='sam3d_ros2_bridge',
        executable='action_server',
        name='sam3d_action_server',
        output='screen',
        parameters=[{
            'zmq_host': LaunchConfiguration('zmq_host'),
            'zmq_port': LaunchConfiguration('zmq_port'),
        }]
    )

    return LaunchDescription([
        zmq_host_arg,
        zmq_port_arg,
        action_server_node
    ])
