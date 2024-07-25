#!/usr/bin/python3

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_prefix


ARGUMENTS = [
    DeclareLaunchArgument('use_gazebo_gui', default_value='true',
                          choices=['true', 'false'],
                          description='Whether or not to launch gzclient'),
    DeclareLaunchArgument('world_path',
          default_value=[os.path.join(get_package_share_directory('drl_agent_bringup'), 'worlds', 'TD3.world')],
          description='SDF world file'),
]


def generate_launch_description():

    # Get packages description and directory
    drl_agent_bringup_package_name = 'drl_agent_bringup'
    drl_agent_description_package_name = 'drl_agent_description'
    velodyne_description_package_name = 'velodyne_description'
    drl_agent_bringup_package_directory = get_package_share_directory(drl_agent_bringup_package_name)
    drl_agent_description_package_directory = get_package_share_directory(drl_agent_description_package_name)
    velodyne_description_package_directory = get_package_share_directory(velodyne_description_package_name)

    # Set the Path to mesh models. NOTE: should be done before gazebo is 1st launched.
    drl_agent_install_dir_path = (get_package_prefix(drl_agent_description_package_name) + '/share')
    velodyne_description_install_dir_path = (get_package_prefix(velodyne_description_package_name) + '/share')

    robot_meshes_path = os.path.join(drl_agent_description_package_directory, 'meshes')
    velodyne_description_meshes_path = os.path.join(velodyne_description_package_directory, 'meshes')
    pkg_models_path = os.path.join(drl_agent_description_package_directory, 'models') # add local models path
    
    gazebo_resource_paths = [drl_agent_install_dir_path, robot_meshes_path, pkg_models_path, velodyne_description_install_dir_path, velodyne_description_meshes_path]
    if 'GAZEBO_MODEL_PATH' in os.environ:
        for resource_path in gazebo_resource_paths:
            if resource_path not in os.environ['GAZEBO_MODEL_PATH']:
                os.environ['GAZEBO_MODEL_PATH'] += (':' + resource_path)
    else:
        os.environ['GAZEBO_MODEL_PATH'] = (':'.join(gazebo_resource_paths))

    border = "+" + "-" * 80 + "+"
    print(border)
    print('> GAZEBO MODELS PATH: ')
    print(str(os.environ['GAZEBO_MODEL_PATH']))
    # print(border)
    # print('> GAZEBO PLUGINS PATH\n'+'='*21)
    # print(str(os.environ['GAZEBO_PLUGIN_PATH']))
    print(border)

    # Launch configurations
    world_path = LaunchConfiguration('world_path')
    use_gazebo_gui = LaunchConfiguration('use_gazebo_gui')

    # Gazebo server
    gazebo_params_yaml_file = os.path.join(
        drl_agent_bringup_package_directory, 'config', 'gazebo_params.yaml')
    gzserver = ExecuteProcess(
        cmd=['gzserver',
             '-s', 'libgazebo_ros_init.so',
             '-s', 'libgazebo_ros_factory.so',
             world_path,
             'extra-gazebo-args', '--ros-args', '--params-file', gazebo_params_yaml_file],
        output='screen',
    )

    # Gazebo client
    gzclient = ExecuteProcess(
        cmd=['gzclient'],
        output='screen',
        condition=IfCondition(use_gazebo_gui),
    )

    ld = LaunchDescription(ARGUMENTS)
    ld.add_action(gzserver)
    ld.add_action(gzclient)

    return ld
