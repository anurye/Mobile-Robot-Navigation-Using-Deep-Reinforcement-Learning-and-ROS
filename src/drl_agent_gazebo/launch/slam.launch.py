#!/usr/bin/python3

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node, PushRosNamespace
from nav2_common.launch import RewrittenYaml
from ament_index_python.packages import get_package_share_directory


ARGUMENTS = [
    DeclareLaunchArgument(
        "use_sim_time",
        default_value="true",
        choices=["true", "false"],
        description="Use sim time",
    ),
    DeclareLaunchArgument(
        "sync",
        default_value="false",
        choices=["true", "false"],
        description="Use synchronous SLAM",
    ),
    DeclareLaunchArgument("namespace", default_value="", description="Robot namespace"),
]


def generate_launch_description():
    drl_agent_gazebo_pkg = get_package_share_directory("drl_agent_gazebo")

    namespace = LaunchConfiguration("namespace")
    sync = LaunchConfiguration("sync")

    slam_params_arg = DeclareLaunchArgument(
        "params",
        default_value=PathJoinSubstitution(
            [drl_agent_gazebo_pkg, "config", "slam.yaml"]
        ),
        description="Robot namespace",
    )

    slam_params = RewrittenYaml(
        source_file=LaunchConfiguration("params"),
        root_key=namespace,
        param_rewrites={},
        convert_types=True,
    )

    remappings = [
        ("/tf", "tf"),
        ("/tf_static", "tf_static"),
        ("/scan", "front_laser/scan"),
        ("/map", "map"),
        ("/map_metadata", "map_metadata"),
    ]

    slam = GroupAction(
        [
            PushRosNamespace(namespace),
            Node(
                package="slam_toolbox",
                executable="sync_slam_toolbox_node",
                name="slam_toolbox",
                output="screen",
                parameters=[
                    slam_params,
                    {"use_sim_time": LaunchConfiguration("use_sim_time")},
                ],
                remappings=remappings,
                condition=IfCondition(sync),
            ),
            Node(
                package="slam_toolbox",
                executable="async_slam_toolbox_node",
                name="slam_toolbox",
                output="screen",
                parameters=[
                    slam_params,
                    {"use_sim_time": LaunchConfiguration("use_sim_time")},
                ],
                remappings=remappings,
                condition=UnlessCondition(sync),
            ),
        ]
    )

    ld = LaunchDescription(ARGUMENTS)
    ld.add_action(slam_params_arg)
    ld.add_action(slam)
    return ld
