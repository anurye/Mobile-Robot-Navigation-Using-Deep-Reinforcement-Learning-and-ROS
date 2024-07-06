#!/usr/bin/python3

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_prefix

def generate_launch_description():

    # Get packages description and directory
    drl_agent_description_package_name = 'drl_agent_description'
    velodyne_description_package_name = 'velodyne_description'
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

    print('> GAZEBO MODELS PATH\n'+'='*20)
    print(str(os.environ['GAZEBO_MODEL_PATH']))
    # print('> GAZEBO PLUGINS PATH\n'+'='*21)
    # print(str(os.environ['GAZEBO_PLUGIN_PATH']))

    # Gazebo launch
    pkg_gazebo_ros = get_package_share_directory('gazebo_ros')
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_gazebo_ros, 'launch', 'gazebo.launch.py'),
        )
    )    

    return LaunchDescription([
        DeclareLaunchArgument(
          'world',
          default_value=[os.path.join(drl_agent_description_package_directory, 'worlds', 'TD3.world'), ''],
          description='SDF world file'),
        gazebo
    ])
