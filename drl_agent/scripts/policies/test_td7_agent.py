#!/usr/bin/env python3

import os
import sys
import argparse
from datetime import date

import torch
import numpy as np

import rclpy
from rclpy.node import Node
from td7_agent import Agent
from environment_interface import EnvInterface
	

class TestTD7(Node):
	def __init__(self, args, env):
		super().__init__("test_td7_agent")
		self.env = env
		self.args = args

		# Get the saved model
		drl_agent_pkg_dir_env = "DRL_AGENT_PACKAGE_PATH"  # echo 'export DRL_AGENT_PACKAGE_PATH=~/drl_agent_ws/src/drl_agent' >> ~/.bashrc
		drl_agent_pkg_dir = os.getenv(drl_agent_pkg_dir_env)
		if drl_agent_pkg_dir is None:
			self.get_logger().error(f"Environment variable: {drl_agent_pkg_dir_env} is not set")
		self.pytorch_models_dir = os.path.join(drl_agent_pkg_dir, "scripts", "pytorch_models")

		state_dim, action_dim, max_action = self.env.get_dimensions()

		# Initialize the agent
		self.rl_agent = Agent(state_dim=state_dim, action_dim=action_dim, max_action=max_action)
		try:
			self.rl_agent.load(self.pytorch_models_dir, self.args.file_name)
			self.get_logger().info(f'{"Model parameters loaded successfuly":-^50}')
		except Exception as e:
			self.get_logger().error(f'{"Could not load network parameters :(":-^50}')
			sys.exit(-1)
		
		# Set seed for the test environment
		self.env.set_env_seed(self.args.seed)
		
		self.done_testing = False
	
	def test(self):
		"""Run continious testing loop"""
		done = False
		episode_timesteps = 0
		state = self.env.reset()

		while True:
			action = self.rl_agent.select_action(np.array(state), self.args.use_checkpoints, use_exploration=False)
			next_state, reward, done, target = self.env.step(action)
			done = 1 if episode_timesteps + 1 == self.args.max_episode_steps else int(done)
			
			# On termination of episode
			if done:
				state = self.env.reset()
				done = False
				episode_timesteps = 0
			else:
				state = next_state
				episode_timesteps += 1
		
		# self.done_testing = True


def main(arguments):
	# Initialize ROS2 communication
	rclpy.init(args=None)
	# Initialize the train_td3_node object
	# Create the environment interface node
	env_interface = EnvInterface()
	test_td7 = TestTD7(arguments, env_interface)
	# Start testing
	test_td7.test()

	try:
		while rclpy.ok() and not test_td7.done_testing:
			rclpy.spin_once(test_td7)
	except KeyboardInterrupt as e:
		pass
	finally:
		test_td7.get_logger().info(f'\n\t{" Testing is Done ":-^35}\n')
		test_td7.get_logger().info("rclpy, shutingdown...")
		test_td7.destroy_node()
		rclpy.shutdown()


if __name__=="__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--seed", default=0, type=int)
	parser.add_argument("--file_name", default=None)
	parser.add_argument("--max_episode_steps", default=500, type=int)
	parser.add_argument("--use_checkpoints", default=True, action=argparse.BooleanOptionalAction)
	arguments = parser.parse_args()

	# File name to save results
	date = "20240720" # date.today().strftime("%Y%m%d")
	if arguments.file_name is None:
		arguments.file_name = f"td7_agent_seed_{arguments.seed}_date_{date}"
	
	# Seed for reproducibility
	torch.manual_seed(arguments.seed)
	np.random.seed(arguments.seed)

	# Start the testing
	main(arguments)

