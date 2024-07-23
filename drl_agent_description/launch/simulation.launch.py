#!/usr/bin/python3

from launch import LaunchDescription
from launch.conditions import IfCondition
from launch.actions import DeclareLaunchArgument
from launch.actions import IncludeLaunchDescription
from launch.substitutions import PathJoinSubstitution
from launch.substitutions.launch_configuration import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python import get_package_share_directory


ARGUMENTS = [
    DeclareLaunchArgument('use_sim_time', default_value='true',
                          choices=['true', 'false'],
                          description='use_sim_time'),
    DeclareLaunchArgument('rviz', default_value='true',
                          choices=['true', 'false'],
                          description='Start rviz.'),
    DeclareLaunchArgument('namespace', default_value='',
                          description='Robot namespace'),
    DeclareLaunchArgument('localization', default_value='false',
                          choices=['true', 'false'],
                          description='Whether to launch localization'),
    DeclareLaunchArgument('slam', default_value='false',
                          choices=['true', 'false'],
                          description='Whether to launch SLAM'),
    DeclareLaunchArgument('nav2', default_value='false',
                          choices=['true', 'false'],
                          description='Whether to launch Nav2'),
]


def generate_launch_description():
    # Get the agent_description share directory
    drl_agent_pkg = 'drl_agent'
    drl_agent_description_pkg = 'drl_agent_description'
    drl_agent_pkg_share = get_package_share_directory(drl_agent_pkg)
    drl_agent_description_share = get_package_share_directory(drl_agent_description_pkg)

    # Paths
    agent_description_launch = PathJoinSubstitution(
        [drl_agent_description_share, 'launch', 'agent_description.launch.py'])
    agent_gazebo_world_launch = PathJoinSubstitution(
        [drl_agent_description_share, 'launch', 'gazebo_world.launch.py'])
    agent_spawn_launch = PathJoinSubstitution(
        [drl_agent_description_share, 'launch', 'spawn_agent.launch.py'])
    rviz_launch = PathJoinSubstitution(
        [drl_agent_description_share, 'launch', 'rviz.launch.py'])
    localization_launch = PathJoinSubstitution(
        [drl_agent_pkg_share, 'launch', 'localization.launch.py'])
    slam_launch = PathJoinSubstitution(
        [drl_agent_pkg_share, 'launch', 'slam.launch.py'])
    nav2_launch = PathJoinSubstitution(
        [drl_agent_pkg_share, 'launch', 'nav2.launch.py'])

    # Launch configurations
    namespace = LaunchConfiguration('namespace')
    use_sim_time = LaunchConfiguration('use_sim_time')

    # agent description
    agent_description = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([agent_description_launch]),
        launch_arguments=[
            ('namespace', namespace),
            ('use_sim_time', use_sim_time)
        ]
    )

    # Gazebo world
    agent_gazebo_world = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([agent_gazebo_world_launch])
    )

    # Agent spawn
    agent_spawn = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([agent_spawn_launch]) # TODO: add location as argument
    )

    # Rviz
    rviz2 = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([rviz_launch]),
        launch_arguments=[
            ('namespace', namespace),
            ('use_sim_time', use_sim_time)
        ],
        condition=IfCondition(LaunchConfiguration('rviz'))
    )

    # Localization
    localization = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([localization_launch]),
        launch_arguments=[
            ('namespace', namespace),
            ('use_sim_time', use_sim_time)
        ],
        condition=IfCondition(LaunchConfiguration('localization'))
    )

    # SLAM
    slam = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([slam_launch]),
        launch_arguments=[
            ('namespace', namespace),
            ('use_sim_time', use_sim_time)
        ],
        condition=IfCondition(LaunchConfiguration('slam'))
    )

    # Nav2
    nav2 = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([nav2_launch]),
        launch_arguments=[
            ('namespace', namespace),
            ('use_sim_time', use_sim_time)
        ],
        condition=IfCondition(LaunchConfiguration('nav2'))
    )

    ld = LaunchDescription(ARGUMENTS)
    ld.add_action(agent_description)
    ld.add_action(agent_gazebo_world)
    ld.add_action(agent_spawn)
    ld.add_action(rviz2)
    ld.add_action(localization)
    ld.add_action(slam)
    ld.add_action(nav2)

    return ld