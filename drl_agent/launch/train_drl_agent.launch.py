#!/usr/bin/env python3

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

def generate_launch_description():
    pkg_drl_launch_description = get_package_share_directory('drl_agent_description')

    # Get simulation launch description from derl_agent_description pkg
    simulation_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_drl_launch_description, 'launch', 'simulation.launch.py')
        ),
    )

    gym_env_node = Node(
        package='drl_agent',
        executable='gazebo_env.py',
        name='gazebo_env',
        output='screen'
    )

    train_td3_agent_node = Node(
        package='drl_agent',
        executable='train_td3_agent.py',
        name='train_td3_agent',
        output='screen'
    )

    return LaunchDescription([
        simulation_launch,
        gym_env_node,
        train_td3_agent_node,
    ])

