#!/usr/bin/python3

from launch_ros.actions import Node
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import PathJoinSubstitution
from launch.substitutions.launch_configuration import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory

ARGUMENTS = [
    DeclareLaunchArgument('use_sim_time', default_value='true',
                          choices=['true', 'false'],
                          description='use_sim_time'),
]

def generate_launch_description():

    # Get rviz config file
    drl_agent_description_pkg = get_package_share_directory('drl_agent_description')
    rviz_config_file = PathJoinSubstitution([drl_agent_description_pkg, 'rviz', 'config.rviz'])

    # Position and orientation
    # [X, Y, Z]
    position = [0.0, 0.0, 0.01]
    # [Roll, Pitch, Yaw]
    orientation = [0.0, 0.0, 0.0]
    # Base Name or robot
    entity_name = "pioneer_3dx"

    # Spawn agent node
    spawn_robot = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        name='spawn_entity',
        output='screen',
        arguments=['-entity',
                   entity_name,
                   '-x', str(position[0]), '-y', str(position[1]
                                                     ), '-z', str(position[2]),
                   '-R', str(orientation[0]), '-P', str(orientation[1]
                                                        ), '-Y', str(orientation[2]),
                   '-topic', '/robot_description'
                   ]
    )

    # rviz node
    rviz2 = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', rviz_config_file],
        parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}],
        remappings=[
            # ('/tf', 'tf'),
            # ('/tf_static', 'tf_static')
        ],
    )

    # create and return launch description object
    ld = LaunchDescription(ARGUMENTS)
    ld.add_action(spawn_robot)
    ld.add_action(rviz2)
    return ld
