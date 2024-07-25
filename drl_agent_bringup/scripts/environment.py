#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import math
import random
import time
import numpy as np
import point_cloud2 as pc2
from squaternion import Quaternion

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup

from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import Marker, MarkerArray
from gazebo_msgs.msg import EntityState

from std_srvs.srv import Empty
from gazebo_msgs.srv import SetEntityState
from drl_agent_interfaces.srv import Step, Reset, Seed, GetDimensions, SampleActionSpace


GOAL_THRESHOLD = 0.3
COLLISION_THRESHOLD = 0.3
TIME_DELTA = 0.1


class Environment(Node):
    """Environment Node for providing services required for DRL.

    This class provides functionalities to interact with an environment through ROS2 services.
    The services include:
    - step: Take an action and get the resulting situation from the environment.
    - reset: Reset the environment and get initial observation.
    - get_dimensions: Get the dimensions of the state, action, and maximum action value.
    """

    def __init__(self, args):
        super().__init__("gym_node")

        self.sensors_callback_group = ReentrantCallbackGroup()  # For sensor callbacks
        self.clients_callback_group = MutuallyExclusiveCallbackGroup()  # For clients
        # Define the dimensions of the state, action, and maximum action value
        self.args = args
        self.environment_dim = self.args.environment_dim
        self.agent_dim = self.args.agent_dim
        self.action_dim = self.args.action_dim
        self.max_action = self.args.max_action

        self.odom_x = 0.0
        self.odom_y = 0.0

        self.goal_x = 1.0
        self.goal_y = 0.0

        self.upper = 5.0
        self.lower = -5.0
        self.velodyne_data = np.ones(self.environment_dim) * 10
        self.last_odom = None

        self.set_agent_state = EntityState()
        self.set_agent_state.name = "pioneer_3dx"
        self.set_box_state = EntityState()

        # Define bens for grouping the velodyne_points
        self.gaps = [[-np.pi / 2 - 0.03, -np.pi / 2 + np.pi / self.environment_dim]]
        for m in range(self.environment_dim - 1):
            self.gaps.append(
                [self.gaps[m][1], self.gaps[m][1] + np.pi / self.environment_dim]
            )
        self.gaps[-1][-1] += 0.03

        # Initialize publishers
        self.vel_pub = self.create_publisher(Twist, "/cmd_vel", 1)
        self.publisher = self.create_publisher(MarkerArray, "goal_point", 3)
        self.publisher2 = self.create_publisher(MarkerArray, "linear_velocity", 1)
        self.publisher3 = self.create_publisher(MarkerArray, "angular_velocity", 1)

        # Initialize subscribers
        qos_profile = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        self.velodyne = self.create_subscription(
            PointCloud2,
            "/velodyne_points",
            self.velodyne_callback,
            qos_profile,
            callback_group=self.sensors_callback_group,
        )
        self.velodyne
        self.odom = self.create_subscription(
            Odometry,
            "/odom",
            self.odom_callback,
            qos_profile,
            callback_group=self.sensors_callback_group,
        )
        self.odom

        # Create services
        self.srv_step = self.create_service(Step, "step", self.step_callback)
        self.srv_reset = self.create_service(Reset, "reset", self.reset_callback)
        self.srv_dimentions = self.create_service(
            GetDimensions, "get_dimensions", self.get_dimensions_callback
        )
        self.srv_seed = self.create_service(Seed, "seed", self.seed_callback)
        self.srv_action_space_sample = self.create_service(
            SampleActionSpace, "action_space_sample", self.sample_action_callback
        )

        # Initialize clients
        self.set_model_state = self.create_client(
            SetEntityState,
            "gazebo/set_entity_state",
            callback_group=self.clients_callback_group,
        )
        self.unpause = self.create_client(Empty, "/unpause_physics")
        self.pause = self.create_client(Empty, "/pause_physics")
        self.reset_proxy = self.create_client(Empty, "/reset_world")

        # Service requests
        self.set_agent_state_req = SetEntityState.Request()
        self.set_static_obs_state_req = SetEntityState.Request()

    def velodyne_callback(self, v):
        """Updates velodyne data

        Reads velodyne point cloud data, converts it into distance data, and
        selects the minimum value for each angle range as a state representation.
        """
        data = list(pc2.read_points(v, skip_nans=False, field_names=("x", "y", "z")))
        self.velodyne_data = np.ones(self.environment_dim) * 10
        for i in range(len(data)):
            if data[i][2] > -0.2:
                dot = data[i][0] * 1 + data[i][1] * 0
                mag1 = math.sqrt(math.pow(data[i][0], 2) + math.pow(data[i][1], 2))
                mag2 = math.sqrt(math.pow(1, 2) + math.pow(0, 2))
                beta = math.acos(dot / (mag1 * mag2)) * np.sign(data[i][1])
                dist = math.sqrt(data[i][0] ** 2 + data[i][1] ** 2 + data[i][2] ** 2)

                for j in range(len(self.gaps)):
                    if self.gaps[j][0] <= beta < self.gaps[j][1]:
                        self.velodyne_data[j] = min(self.velodyne_data[j], dist)
                        break

    def odom_callback(self, od_data):
        """Updates the latest odometry data"""
        self.last_odom = od_data

    def seed_callback(self, request, response):
        """Sets environment seed for reproducibility of the training process."""
        np.random.seed(request.seed)
        response.success = True
        return response

    def sample_action_callback(self, request, response):
        """Samples an action from the action space."""
        action = np.random.uniform(self.args.actions_low, self.args.actions_high)
        response.action = np.array(action, dtype=np.float32).tolist()
        return response

    def get_dimensions_callback(self, request, response):
        """Returns the dimensions of the state, action, and maximum action value"""
        response.state_dim = self.environment_dim + self.agent_dim
        response.action_dim = self.action_dim
        response.max_action = self.max_action
        return response

    def step_callback(self, request, response):
        """Executes a step in the environment, updating the robot's state and reading the new state.

        This involves publishing the robot action, unpausing the simulation, propagating the state for a set time interval,
        pausing the simulation, reading laser data to detect collisions, calculating the robot heading and distance to the goal,
        and checking if the goal is reached and calculating the reward.
        """
        action = request.action
        target = False

        # Publish the robot action
        vel_cmd = Twist()
        vel_cmd.linear.x = float(action[0])
        vel_cmd.angular.z = float(action[1])
        self.vel_pub.publish(vel_cmd)
        self.publish_markers(action)

        while not self.unpause.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Service /unpause_physics not available, waiting again...")
        try:
            self.unpause.call_async(Empty.Request())
        except Exception as e:
            self.get_logger().error("/unpause_physics service call failed: %s" % str(e))

        # propagate state for TIME_DELTA seconds
        time.sleep(TIME_DELTA)

        while not self.pause.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Service /pause_physics not available, waiting again...")
        try:
            pass
            self.pause.call_async(Empty.Request())
        except Exception as e:
            self.get_logger().error("/pause_physics service call failed: %s" % str(e))

        # read velodyne laser state
        done, collision, min_laser = self.observe_collision(self.velodyne_data)
        v_state = []
        v_state[:] = self.velodyne_data[:]
        laser_state = [v_state]

        # Calculate robot heading from odometry data
        self.odom_x = self.last_odom.pose.pose.position.x
        self.odom_y = self.last_odom.pose.pose.position.y
        quaternion = Quaternion(
            self.last_odom.pose.pose.orientation.w,
            self.last_odom.pose.pose.orientation.x,
            self.last_odom.pose.pose.orientation.y,
            self.last_odom.pose.pose.orientation.z,
        )
        euler = quaternion.to_euler(degrees=False)
        angle = round(euler[2], 4)

        # Calculate distance to the goal from the robot
        distance = np.linalg.norm(
            [self.odom_x - self.goal_x, self.odom_y - self.goal_y]
        )

        # Calculate the relative angle between the robots heading and heading toward the goal
        skew_x = self.goal_x - self.odom_x
        skew_y = self.goal_y - self.odom_y
        dot = skew_x * 1 + skew_y * 0
        mag1 = math.sqrt(math.pow(skew_x, 2) + math.pow(skew_y, 2))
        mag2 = math.sqrt(math.pow(1, 2) + math.pow(0, 2))
        beta = math.acos(dot / (mag1 * mag2))
        if skew_y < 0:
            if skew_x < 0:
                beta = -beta
            else:
                beta = 0 - beta
        theta = beta - angle
        if theta > np.pi:
            theta = np.pi - theta
            theta = -np.pi - theta
        if theta < -np.pi:
            theta = -np.pi - theta
            theta = np.pi - theta

        # Detect if the goal has been reached and give a large positive reward
        if distance < GOAL_THRESHOLD:
            self.get_logger().info("+===-----------------===+")
            self.get_logger().info("|     GOAL REACHED      |")
            self.get_logger().info("+===-----------------===+")
            target = True
            done = True

        robot_state = [distance, theta, action[0], action[1]]
        state = np.append(laser_state, robot_state)
        reward = self.get_reward(target, collision, action, min_laser)

        response.state = state.tolist()
        response.reward = reward
        response.done = done
        response.target = target

        return response

    def reset_callback(self, request, response):
        """Resets the state of the environment and returns an initial observation, state"""

        while not self.reset_proxy.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Service /reset_world not available, waiting again...")
        try:
            self.reset_proxy.call_async(Empty.Request())
        except Exception as e:
            self.get_logger().error("/reset_world service call failed: %s" % str(e))

        angle = np.random.uniform(-np.pi, np.pi)
        quaternion = Quaternion.from_euler(0.0, 0.0, angle)
        object_state = self.set_agent_state

        x = 0.0
        y = 0.0
        position_ok = False
        while not position_ok:
            x = np.random.uniform(-4.5, 4.5)
            y = np.random.uniform(-4.5, 4.5)
            position_ok = self.check_pos(x, y)
        object_state.pose.position.x = x
        object_state.pose.position.y = y
        object_state.pose.position.z = 0.0
        object_state.pose.orientation.x = quaternion.x
        object_state.pose.orientation.y = quaternion.y
        object_state.pose.orientation.z = quaternion.z
        object_state.pose.orientation.w = quaternion.w

        self.set_agent_state_req._state = object_state

        while not self.set_model_state.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Service /gazebo/set_entity_state not available, waiting again...")
        try:
            _ = self.set_model_state.call_async(self.set_agent_state_req)
        except Exception as e:
            self.get_logger().error("/gazebo/set_entity_state service call failed: %s" % str(e))

        self.odom_x = object_state.pose.position.x
        self.odom_y = object_state.pose.position.y

        # set a random goal in empty space in environment
        self.change_goal()
        # randomly scatter boxes in the environment
        self.random_box()
        self.publish_markers([0.0, 0.0])

        while not self.unpause.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Service /unpause_physics not available, waiting again...")
        try:
            self.unpause.call_async(Empty.Request())
        except Exception as e:
            self.get_logger().error("/unpause_physics service call failed: %s" % str(e))

        time.sleep(TIME_DELTA)

        while not self.pause.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Service /pause_physics not available, waiting again...")
        try:
            self.pause.call_async(Empty.Request())
        except Exception as e:
            self.get_logger().error("/pause_physics service call failed: %s" % str(e))

        v_state = []
        v_state[:] = self.velodyne_data[:]
        laser_state = [v_state]

        distance = np.linalg.norm(
            [self.odom_x - self.goal_x, self.odom_y - self.goal_y]
        )

        skew_x = self.goal_x - self.odom_x
        skew_y = self.goal_y - self.odom_y

        dot = skew_x * 1 + skew_y * 0
        mag1 = math.sqrt(math.pow(skew_x, 2) + math.pow(skew_y, 2))
        mag2 = math.sqrt(math.pow(1, 2) + math.pow(0, 2))
        beta = math.acos(dot / (mag1 * mag2))

        if skew_y < 0:
            if skew_x < 0:
                beta = -beta
            else:
                beta = 0 - beta
        theta = beta - angle

        if theta > np.pi:
            theta = np.pi - theta
            theta = -np.pi - theta
        if theta < -np.pi:
            theta = -np.pi - theta
            theta = np.pi - theta

        robot_state = [distance, theta, 0.0, 0.0]
        state = np.append(laser_state, robot_state)

        response.state = state.tolist()
        return response

    def change_goal(self):
        """Places a new goal and ensures its location is not on one of the obstacles"""
        goal_ok = False

        if self.upper < 10:
            self.upper += 0.004
        if self.lower > -10:
            self.lower -= 0.004

        while not goal_ok:
            self.goal_x = self.odom_x + random.uniform(self.upper, self.lower)
            self.goal_y = self.odom_y + random.uniform(self.upper, self.lower)
            goal_ok = self.check_pos(self.goal_x, self.goal_y)

    def random_box(self):
        """Randomly changes the location of the boxes in the environment on each reset to randomize the training environment"""
        for i in range(4):
            box_state = self.set_box_state
            box_state.name = "cardboard_box_" + str(i)

            x = 0.0
            y = 0.0
            box_ok = False
            while not box_ok:
                x = np.random.uniform(-6, 6)
                y = np.random.uniform(-6, 6)
                box_ok = self.check_pos(x, y)
                distance_to_robot = np.linalg.norm([x - self.odom_x, y - self.odom_y])
                distance_to_goal = np.linalg.norm([x - self.goal_x, y - self.goal_y])
                if distance_to_robot < 1.5 or distance_to_goal < 1.5:
                    box_ok = False
            box_state.pose.position.x = x
            box_state.pose.position.y = y
            box_state.pose.position.z = 0.0
            box_state.pose.orientation.x = 0.0
            box_state.pose.orientation.y = 0.0
            box_state.pose.orientation.z = 0.0
            box_state.pose.orientation.w = 1.0

            self.set_static_obs_state_req._state = box_state

            while not self.set_model_state.wait_for_service(timeout_sec=1.0):
                self.get_logger().info("Service /gazebo/set_entity_state not available, waiting again...")
            try:
                _ = self.set_model_state.call_async(self.set_static_obs_state_req)
            except Exception as e:
                self.get_logger().error("/gazebo/set_entity_state service call failed: %s" % str(e))

    def publish_markers(self, action):
        """Publishes visual data for Rviz to visualize the goal and the robot's actions"""
        markerArray = MarkerArray()
        marker = Marker()
        marker.header.frame_id = "odom"
        marker.type = marker.CYLINDER
        marker.action = marker.ADD
        marker.scale.x = 0.1
        marker.scale.y = 0.1
        marker.scale.z = 0.01
        marker.color.a = 1.0
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.pose.orientation.w = 1.0
        marker.pose.position.x = self.goal_x
        marker.pose.position.y = self.goal_y
        marker.pose.position.z = 0.0
        markerArray.markers.append(marker)
        self.publisher.publish(markerArray)

        markerArray2 = MarkerArray()
        marker2 = Marker()
        marker2.header.frame_id = "odom"
        marker2.type = marker.CUBE
        marker2.action = marker.ADD
        marker2.scale.x = abs(action[0])
        marker2.scale.y = 0.1
        marker2.scale.z = 0.01
        marker2.color.a = 1.0
        marker2.color.r = 1.0
        marker2.color.g = 0.0
        marker2.color.b = 0.0
        marker2.pose.orientation.w = 1.0
        marker2.pose.position.x = 5.0
        marker2.pose.position.y = 0.0
        marker2.pose.position.z = 0.0
        markerArray2.markers.append(marker2)
        self.publisher2.publish(markerArray2)

        markerArray3 = MarkerArray()
        marker3 = Marker()
        marker3.header.frame_id = "odom"
        marker3.type = marker.CUBE
        marker3.action = marker.ADD
        marker3.scale.x = abs(action[1])
        marker3.scale.y = 0.1
        marker3.scale.z = 0.01
        marker3.color.a = 1.0
        marker3.color.r = 1.0
        marker3.color.g = 0.0
        marker3.color.b = 0.0
        marker3.pose.orientation.w = 1.0
        marker3.pose.position.x = 5.0
        marker3.pose.position.y = 0.2
        marker3.pose.position.z = 0.0
        markerArray3.markers.append(marker3)
        self.publisher3.publish(markerArray3)

    @staticmethod
    def observe_collision(laser_data):
        """Detect a collision from laser data.

        Returns
        -------
        tuple
            A tuple containing:
            done : bool
                True if a collision is detected/goal reached, otherwise False.
            collision : bool
                True if a collision is detected
            min_laser : float
                The minimum laser reading.
        """
        min_laser = min(laser_data)
        if min_laser < COLLISION_THRESHOLD:
            return True, True, min_laser
        return False, False, min_laser

    @staticmethod
    def get_reward(target, collision, action, min_laser):
        """Calculate the reward based on the current state and action taken"""
        if target:
            return 100.0
        if collision:
            return -100.0
        obstacle_reward = (min_laser - 1)/2 if min_laser < 1.0 else 0.0
        action_reward = action[0]/2 - abs(action[1])/2 - 0.001
        return action_reward + obstacle_reward

    @staticmethod
    def check_pos(x, y):
        """Check if the proposed pos is located in unoccupied zone"""
        goal_ok = True
        if -3.8 > x > -6.2 and 6.2 > y > 3.8:
            goal_ok = False
        if -1.3 > x > -2.7 and 4.7 > y > -0.2:
            goal_ok = False
        if -0.3 > x > -4.2 and 2.7 > y > 1.3:
            goal_ok = False
        if -0.8 > x > -4.2 and -2.3 > y > -4.2:
            goal_ok = False
        if -1.3 > x > -3.7 and -0.8 > y > -2.7:
            goal_ok = False
        if 4.2 > x > 0.8 and -1.8 > y > -3.2:
            goal_ok = False
        if 4 > x > 2.5 and 0.7 > y > -3.2:
            goal_ok = False
        if 6.2 > x > 3.8 and -3.3 > y > -4.2:
            goal_ok = False
        if 4.2 > x > 1.3 and 3.7 > y > 1.5:
            goal_ok = False
        if -3.0 > x > -7.2 and 0.5 > y > -1.5:
            goal_ok = False
        if x > 4.5 or x < -4.5 or y > 4.5 or y < -4.5:
            goal_ok = False
        return goal_ok


def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--environment_dim", type=int, default=20)
    parser.add_argument("--agent_dim", type=int, default=4)
    parser.add_argument("--action_dim", type=int, default=2)
    parser.add_argument("--max_action", type=float, default=1.0)
    parser.add_argument("--actions_low", nargs=2, type=float, default=[-1, -1.0])
    parser.add_argument("--actions_high", nargs=2, type=float, default=[1.0, 1.0])
    arguments = parser.parse_args()

    # Initialize the ROS2 communication
    rclpy.init(args=args)
    # Create the environment node
    env = Environment(arguments)
    # Use MultiThreadedExecutor to handle the two sensor callbacks in parallel.
    executor = MultiThreadedExecutor(num_threads=3)
    executor.add_node(env)
    try:
        executor.spin()
    finally:
        env.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
