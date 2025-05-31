#!/usr/bin/python3

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import Command, PathJoinSubstitution
from launch.substitutions.launch_configuration import LaunchConfiguration

from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


ARGUMENTS = [
    DeclareLaunchArgument(
        "use_sim_time",
        default_value="true",
        choices=["true", "false"],
        description="use_sim_time",
    ),
    DeclareLaunchArgument("namespace", default_value="", description="Robot namespace"),
    DeclareLaunchArgument(
        "launch_joint_state_pub",
        default_value="false",
        description="Whether or not to launch joint-state-publisher",
    ),
]


def generate_launch_description():
    drl_agent_description_pkg = get_package_share_directory("drl_agent_description")
    xacro_file = PathJoinSubstitution(
        [drl_agent_description_pkg, "urdf", "p3dx", "pioneer3dx.urdf.xacro"]
    )
    namespace = LaunchConfiguration("namespace")

    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="screen",
        parameters=[
            {"use_sim_time": LaunchConfiguration("use_sim_time")},
            {
                "robot_description": ParameterValue(
                    Command(["xacro", " ", xacro_file, " ", "namespace:=", namespace]),
                    value_type=str,
                )
            },
        ],
        remappings=[("/tf", "tf"), ("/tf_static", "tf_static")],
    )

    joint_state_publisher = Node(
        package="joint_state_publisher",
        executable="joint_state_publisher",
        name="joint_state_publisher",
        output="screen",
        parameters=[{"use_sim_time": LaunchConfiguration("use_sim_time")}],
        remappings=[("/tf", "tf"), ("/tf_static", "tf_static")],
        condition=IfCondition(LaunchConfiguration("launch_joint_state_pub")),
    )

    ld = LaunchDescription(ARGUMENTS)
    ld.add_action(robot_state_publisher)
    ld.add_action(joint_state_publisher)
    return ld
