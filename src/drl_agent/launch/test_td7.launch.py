from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():

    gym_env_node = Node(
        package="drl_agent",
        executable="environment.py",
        name="environment_node",
        output="screen",
        emulate_tty=True,
        parameters=[{"environment_mode": "test"}],
    )

    test_td7_node = Node(
        package="drl_agent",
        executable="test_td7_agent.py",
        name="test_td7_node",
        output="screen",
        emulate_tty=True,
    )

    return LaunchDescription(
        [
            gym_env_node,
            test_td7_node,
        ]
    )
