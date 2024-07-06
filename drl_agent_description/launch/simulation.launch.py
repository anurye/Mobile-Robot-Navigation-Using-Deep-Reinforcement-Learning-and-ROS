import os

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python import get_package_share_directory


def generate_launch_description():
    # Get the agent_description share directory
    agent_description_pkg = 'drl_agent_description'
    agent_description_share = get_package_share_directory(agent_description_pkg)

    # Agent description launch
    agent_description_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(agent_description_share,
                         'launch/agent_description.launch.py')
        )
    )

    # Gazebo world launch
    agent_gazebo_world_launch = IncludeLaunchDescription(
        os.path.join(agent_description_share, 
                     'launch/gazebo_world.launch.py')
    )

    # Spawn agent launch
    agent_spawn_launch = IncludeLaunchDescription(
        os.path.join(agent_description_share, 
                     'launch/spawn_agent.launch.py')
    )

    ld = LaunchDescription()
    ld.add_action(agent_description_launch)
    ld.add_action(agent_gazebo_world_launch)
    ld.add_action(agent_spawn_launch)

    return ld