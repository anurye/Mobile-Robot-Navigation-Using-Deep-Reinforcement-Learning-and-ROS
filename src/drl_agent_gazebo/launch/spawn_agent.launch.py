#!/usr/bin/python3

from launch_ros.actions import Node
from launch import LaunchDescription


def generate_launch_description():
    # Position and orientation
    position = [0.0, 0.0, 0.0]  # [X, Y, Z]
    orientation = [0.0, 0.0, 0.0]  # [Roll, Pitch, Yaw]
    # Robot name
    entity_name = "pioneer_3dx"

    # Spawn agent node
    spawn_robot = Node(
        package="gazebo_ros",
        executable="spawn_entity.py",
        name="spawn_entity",
        output="screen",
        arguments=[
            "-entity",
            entity_name,
            "-x",
            str(position[0]),
            "-y",
            str(position[1]),
            "-z",
            str(position[2]),
            "-R",
            str(orientation[0]),
            "-P",
            str(orientation[1]),
            "-Y",
            str(orientation[2]),
            "-topic",
            "/robot_description",
        ],
    )

    return LaunchDescription(
        [
            spawn_robot,
        ]
    )
