#!/usr/bin/env python3

import os
import sys
import time
import yaml
import json
import torch
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from nav_msgs.msg import Odometry
from td7_agent import Agent
from environment_interface import EnvInterface


class TestTD7(EnvInterface):
	def __init__(self):
		super().__init__("test_td7_agent")

		"""************************************************
		** Test config
		************************************************"""
		# Base directory for loading models and for saving trajectories
		config_file_name = "test_config.yaml"
		drl_agent_src_path_env = "DRL_AGENT_SRC_PATH"
		drl_agent_src_path = os.getenv(drl_agent_src_path_env)
		if drl_agent_src_path is None:
			self.get_logger().error(f"Environment variable: {drl_agent_src_path_env} is not set")
			sys.exit(-1)
		test_config_file_path = os.path.join(drl_agent_src_path, "drl_agent", "config", config_file_name)
		# Model will be loaded from
		self.pytorch_models_dir = os.path.join(drl_agent_src_path, "drl_agent", "pytorch_models")
		# Trajectories will be save in
		self.trajectories_dir = os.path.join(drl_agent_src_path, "drl_agent", "trajectories")
		os.makedirs(self.trajectories_dir, exist_ok=True)
		# Load test config file
		self.test_config = self.load_yaml_file(test_config_file_path)["test_setting"]
		self.seed = self.test_config["seed"]
		save_date = self.test_config["save_date"]
		base_file_name = self.test_config["base_file_name"]
		self.file_name = f"{base_file_name}_seed_{self.seed}_date_{save_date}"
		self.use_checkpoints = self.test_config["use_checkpoints"]
		self.max_episode_steps = self.test_config["max_episode_steps"]
		# Set seed
		torch.manual_seed(self.seed)
		np.random.seed(self.seed)
		self.set_env_seed(self.seed)

		# Initialize the agent
		state_dim, action_dim, max_action = self.get_dimensions()
		self.rl_agent = Agent(state_dim=state_dim, action_dim=action_dim, max_action=max_action)
		try:
			self.rl_agent.load(self.pytorch_models_dir, self.file_name)
			self.get_logger().info(f'{"Model parameters loaded successfuly":-^50}')
		except Exception as e:
			self.get_logger().error(f'{"Could not load network parameters :(":-^50}')
			sys.exit(-1)

		# Callback groups for handling sensors and services in parallel
		self.odom_callback_group = MutuallyExclusiveCallbackGroup() 
		# Initialize subscribers
		qos_profile = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
		self.odom = self.create_subscription(Odometry, "odom", self.odom_callback, 
									   qos_profile, callback_group=self.odom_callback_group)
		self.odom
		# Initialize lists for storing current and all trajectories as well as last odom
		self.last_odom = None
		self.all_trajectories = []
		self.current_trajectory = []

		# Initialize metrics
		self.total_distance_traveled = 0
		self.total_goal_distance = 0
		self.total_time = 0
		self.num_episodes = 0

		self.test()

	def load_yaml_file(self, config_file_path):
		"""Loads test configuration file"""
		try:
			with open(config_file_path, 'r') as file:
				config = yaml.safe_load(file)
		except Exception as e:
			self.get_logger().info(f"Unable to load: {config_file_path}: {e}")
		return config

	def odom_callback(self, od_data):
		"""Updates the latest odometry data"""
		self.last_odom = od_data

	def test(self):
		"""Run continious testing loop"""
		done = False
		episode_timesteps = 0
		state = self.reset()
		# episode_start_time = time.time()
		while True:
			action = self.rl_agent.select_action(np.array(state), self.use_checkpoints, use_exploration=False)
			next_state, reward, done, target = self.step(action)
			done = 1 if episode_timesteps + 1 == self.max_episode_steps else int(done)

			# Record current position
			if not self.last_odom is None:
				x = self.last_odom.pose.pose.position.x
				y = self.last_odom.pose.pose.position.y
				self.current_trajectory.append({'x': x, 'y': y})
			
			# On termination of episode
			if done:
				# Save recorded trajectory and reset current trajectory
				self.all_trajectories.append(self.current_trajectory)
				self.save_trajectories()

				# Calculate metrics
				# episode_distance_traveled = self.calculate_distance(self.current_trajectory)
				# self.total_distance_traveled += episode_distance_traveled
				# goal_distance = np.sqrt(self.current_trajectory[0]['x']**2 + self.current_trajectory[0]['y']**2)
				# self.total_goal_distance += goal_distance
				# episode_time = time.time() - episode_start_time
				# self.total_time += episode_time
				# self.num_episodes += 1
				# Reset
				self.current_trajectory = []
				try:
					state = self.reset()
				except Exception:
					pass
				done = False
				episode_timesteps = 0
				# episode_start_time = time.time()
			else:
				state = next_state
				episode_timesteps += 1
	
	def calculate_distance(self, traj):
		"""Calculates the total distance traveled in a trajectory"""
		distance = 0.0
		for i in range(1, len(traj)):
			x1, y1 = traj[i-1]['x'], traj[i-1]['y']
			x2, y2 = traj[i]['x'], traj[i]['y']
			distance += np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
		return distance
	
	def save_trajectories(self):
		filename = os.path.join(self.trajectories_dir, f"traj_for_seed_{self.seed}.json")
		with open(filename, 'w') as file:
			json.dump(self.all_trajectories, file, indent=4)
		self.get_logger().info(f'Saved trajectory data to {filename}')



def main():
	# Initialize ROS2 communication
	rclpy.init(args=None)
	test_td7 = TestTD7()
	executor = MultiThreadedExecutor(num_threads=2)
	executor.add_node(test_td7)
	try:
		while rclpy.ok():
			executor.spin()
	except KeyboardInterrupt:
		pass
	finally:
		test_td7.get_logger().info(f"{'Testing is Done':-^35}")
		test_td7.get_logger().info("rclpy, shutingdown...")
		test_td7.destroy_node()
		rclpy.shutdown()


if __name__=="__main__":
	main()
