#!/usr/bin/python3

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_prefix


ARGUMENTS = [
    DeclareLaunchArgument(
        "use_gazebo_gui",
        default_value="true",
        choices=["true", "false"],
        description="Whether or not to launch gzclient",
    ),
    DeclareLaunchArgument(
        "world_path",
        default_value=[
            os.path.join(
                get_package_share_directory("drl_agent_gazebo"),
                "worlds",
                "td7_static.world",
            )
        ],
        description="SDF world file",
    ),
]


def generate_launch_description():

    """*************************************************************************************
    ** Get package share directories
    *************************************************************************************"""
    drl_agent_gazebo_package_name = "drl_agent_gazebo"
    drl_agent_description_package_name = "drl_agent_description"
    velodyne_description_package_name = "velodyne_description"
    drl_agent_gazebo_package_directory = get_package_share_directory(
        drl_agent_gazebo_package_name
    )
    drl_agent_description_package_directory = get_package_share_directory(
        drl_agent_description_package_name
    )
    velodyne_description_package_directory = get_package_share_directory(
        velodyne_description_package_name
    )
    gazebo_ros_package_directory = get_package_share_directory("gazebo_ros")

    """*************************************************************************************
    ** Set the Path to mesh models. NOTE: should be done before gazebo is 1st launched.
    *************************************************************************************"""
    drl_agent_gazebo_install_dir_path = (
        get_package_prefix(drl_agent_gazebo_package_name) + "/share"
    )
    drl_agent_install_dir_path = (
        get_package_prefix(drl_agent_description_package_name) + "/share"
    )
    velodyne_description_install_dir_path = (
        get_package_prefix(velodyne_description_package_name) + "/share"
    )

    robot_meshes_path = os.path.join(drl_agent_description_package_directory, "meshes")
    velodyne_description_meshes_path = os.path.join(
        velodyne_description_package_directory, "meshes"
    )
    drl_agent_description_models_path = os.path.join(
        drl_agent_description_package_directory, "models"
    )
    drl_agent_gazebo_models_path = os.path.join(
        drl_agent_gazebo_package_directory, "models"
    )

    gazebo_resource_paths = [
        drl_agent_gazebo_install_dir_path,
        drl_agent_install_dir_path,
        robot_meshes_path,
        drl_agent_description_models_path,
        drl_agent_gazebo_models_path,
        velodyne_description_install_dir_path,
        velodyne_description_meshes_path,
    ]
    if "GAZEBO_MODEL_PATH" in os.environ:
        for resource_path in gazebo_resource_paths:
            if resource_path not in os.environ["GAZEBO_MODEL_PATH"]:
                os.environ["GAZEBO_MODEL_PATH"] += ":" + resource_path
    else:
        os.environ["GAZEBO_MODEL_PATH"] = ":".join(gazebo_resource_paths)

    border = "+" + "-" * 80 + "+"
    print(border)
    print("> GAZEBO MODELS PATH: ")
    print(str(os.environ["GAZEBO_MODEL_PATH"]))
    # print(border)
    # print('> GAZEBO PLUGINS PATH\n'+'='*21)
    # print(str(os.environ['GAZEBO_PLUGIN_PATH']))
    print(border)

    """*************************************************************************************
    Launch configurations and gazebo launch
    *************************************************************************************"""
    world_path = LaunchConfiguration("world_path")
    use_gazebo_gui = LaunchConfiguration("use_gazebo_gui")

    gzserver_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(gazebo_ros_package_directory, "launch", "gzserver.launch.py")
        ),
        launch_arguments=[
            ("world", world_path),
        ],
    )

    gzclient_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(gazebo_ros_package_directory, "launch", "gzclient.launch.py")
        ),
        condition=IfCondition(use_gazebo_gui),
    )

    ld = LaunchDescription(ARGUMENTS)
    ld.add_action(gzserver_cmd)
    ld.add_action(gzclient_cmd)

    return ld
