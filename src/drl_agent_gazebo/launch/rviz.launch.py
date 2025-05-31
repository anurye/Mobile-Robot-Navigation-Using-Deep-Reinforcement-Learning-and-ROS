#!/usr/bin/python3

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


ARGUMENTS = [
    DeclareLaunchArgument(
        "use_sim_time",
        default_value="true",
        choices=["true", "false"],
        description="Use sim time",
    ),
    DeclareLaunchArgument("namespace", default_value="", description="Robot namespace"),
]


def generate_launch_description():
    drl_agent_gazebo_share = get_package_share_directory("drl_agent_gazebo")

    rviz_config_file = PathJoinSubstitution(
        [drl_agent_gazebo_share, "config", "config.rviz"]
    )

    # rviz node
    rviz2 = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        arguments=["-d", rviz_config_file],
        namespace=LaunchConfiguration("namespace"),
        parameters=[{"use_sim_time": LaunchConfiguration("use_sim_time")}],
        remappings=[("/tf", "tf"), ("/tf_static", "tf_static")],
    )

    ld = LaunchDescription(ARGUMENTS)
    ld.add_action(rviz2)

    return ld
