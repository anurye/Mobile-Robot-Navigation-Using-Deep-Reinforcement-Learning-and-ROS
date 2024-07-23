#!/usr/bin/env python3

import os
import time
from datetime import date
import argparse

import rclpy
from rclpy.node import Node

import numpy as np
import torch
from td7_agent import Agent
from environment_interface import EnvInterface


class TrainTD7(Node):
    def __init__(self, args, env):
        super().__init__("train_td7_node")

        # Set arguments
        self.args = args
        self.env = env

        # Setup directories for saving models, results and logs
        drl_agent_pkg_path_env = "DRL_AGENT_PACKAGE_PATH" # echo 'export DRL_AGENT_PACKAGE_PATH=~/drl_agent_ws/src/drl_agent' >> ~/.bashrc
        drl_agent_pkg_path = os.getenv(drl_agent_pkg_path_env)
        if drl_agent_pkg_path is None:
            self.get_logger().error(f"Environment variable: {drl_agent_pkg_path_env}, is not set")
        self.pytorch_models_dir = os.path.join(drl_agent_pkg_path, "scripts", "pytorch_models")
        self.final_models_dir = os.path.join(drl_agent_pkg_path, "scripts", "final_models")
        self.results_dir = os.path.join(drl_agent_pkg_path, "scripts", "results")
        self.log_dir = os.path.join(drl_agent_pkg_path, "scripts", "logs")

        # Create the directories, if they don't exist
        os.makedirs(self.pytorch_models_dir, exist_ok=True)
        os.makedirs(self.final_models_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

        # Set seed for training environment
        self.env.set_env_seed(self.args.seed)
        # Get the dimensions of the environment
        self.state_dim, self.action_dim, self.max_action = self.env.get_dimensions()
        # Maximum number of steps in an episode
        self.max_episode_steps = self.args.max_episode_steps
        # Initialize the agent
        self.rl_agent = Agent(
            self.state_dim,
            self.action_dim,
            self.max_action,
            self.args.offline,
            log_dir=self.log_dir,
        )

        # Try to load the model
        if self.args.load_model:
            try:
                self.rl_agent.load(self.pytorch_models_dir, self.args.file_name)
                self.get_logger().info("Model loaded")
            except Exception as e:
                self.get_logger().warning(f"Failed to load the model: {e}")

        # Flag to indicate that the training is done
        self.done_training = False

        # Log data
        self.border = "+" + "-" * 80 + "+"
        self.get_logger().info(self.border)
        self.get_logger().info(f"| File name: {self.args.file_name}: Seed: {self.args.seed}")
        self.get_logger().info(self.border)
        self.get_logger().info("| Results will be saved in:")
        self.get_logger().info(f"|  {self.pytorch_models_dir}")
        self.get_logger().info(f"|  {self.final_models_dir}")
        self.get_logger().info(f"|  {self.results_dir}")
        self.get_logger().info(f"|  {self.log_dir}")
        self.get_logger().info(self.border)
        self.get_logger().info("| Environment")
        self.get_logger().info(self.border)
        self.get_logger().info(f"| State Dim: {self.state_dim}")
        self.get_logger().info(f"| Action Dim: {self.action_dim}")
        self.get_logger().info(f"| Max Action: {self.max_action}")
        self.get_logger().info(self.border)

    def save_models(self, directory, file_name):
        """Save the models at the given step"""
        self.rl_agent.save(directory, file_name)
        self.get_logger().info("Models updated")

    def train_online(self):
        """Train the agent online"""
        # Initialize the variables
        start_time = time.time()
        evals = []
        epoch = 1
        timesteps_sice_eval = 0
        allow_train = False

        state, ep_finished = self.env.reset(), False
        ep_total_reward, ep_timesteps, ep_num = 0, 0, 1

        for t in range(1, int(self.args.max_timesteps + 1)):

            if allow_train:
                action = self.rl_agent.select_action(np.array(state))
            else:
                action = self.env.sample_action_space()
            
            # Act
            next_state, reward, ep_finished, _ = self.env.step(action)

            ep_total_reward += reward
            ep_timesteps += 1

            done = float(ep_finished) if ep_timesteps < self.max_episode_steps else 0
            self.rl_agent.replay_buffer.add(state, action, next_state, reward, done)

            state = next_state

            if allow_train and not self.args.use_checkpoints:
                self.rl_agent.train()

            if ep_finished or ep_timesteps == self.max_episode_steps:
                self.get_logger().info(f"Total T: {t+1} Episode Num: {ep_num} Episode T: {ep_timesteps} Reward: {ep_total_reward:.3f}")
                if allow_train and self.args.use_checkpoints:
                    self.rl_agent.train_and_checkpoint(ep_timesteps, ep_total_reward)

                if timesteps_sice_eval >= self.args.eval_freq and allow_train:
                    timesteps_sice_eval %= self.args.eval_freq
                    # Save the models
                    self.save_models(self.pytorch_models_dir, self.args.file_name)
                    self.evaluate_and_print(evals, epoch, start_time)
                    epoch += 1
                
                if t >= self.args.timesteps_before_training:
                    allow_train = True

                state, done = self.env.reset(), False
                ep_total_reward, ep_timesteps = 0, 0
                ep_num += 1

            timesteps_sice_eval += 1
        # Indicate that the training is done
        self.done_training = True

    def evaluate_and_print(self, evals, epoch, start_time):
        """Evaluate the agent and print the results"""

        self.get_logger().info(self.border)
        self.get_logger().info(f"| Evaluation at epoch: {epoch}")
        self.get_logger().info(f"| Total time passed: {round((time.time()-start_time)/60.,2)} min(s)")

        total_reward = np.zeros(self.args.eval_eps)
        for ep in range(self.args.eval_eps):
            state, done = self.env.reset(), False
            ep_timesteps = 0
            while not done and ep_timesteps < self.max_episode_steps:
                action = self.rl_agent.select_action(np.array(state), self.args.use_checkpoints, use_exploration=False)
                # Act
                state, reward, done, _ = self.env.step(action)
                total_reward[ep] += reward
                ep_timesteps += 1

        self.get_logger().info(f"| Average reward over {self.args.eval_eps} episodes: {total_reward.mean():.3f}")
        self.get_logger().info(self.border)
        evals.append(total_reward.mean())
        np.save(f"{self.results_dir}/{self.args.file_name}", evals)


def main(arguments):
    # Initialize the ROS2 communication
    rclpy.init(args=None)
    # Create the environment interface node
    env_interface = EnvInterface()
    # Create the node
    train_td7_node = TrainTD7(arguments, env=env_interface)
    # Start training
    train_td7_node.train_online()
    try:
        while rclpy.ok():
            if train_td7_node.done_training:
                train_td7_node.get_logger().info(f"{'Training done':=^30}")
                train_td7_node.save_models(train_td7_node.final_models_dir, arguments.file_name)
                train_td7_node.get_logger().info("Final models saved")
                break
            rclpy.spin_once(train_td7_node)
    except KeyboardInterrupt as e:
        train_td7_node.get_logger().warning(f"KeyboardInterrupt: {e}")
        train_td7_node.rl_agent.save(train_td7_node.final_models_dir, arguments.file_name)
    finally:
        train_td7_node.get_logger().info("rclpy, shutting down...")
        train_td7_node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # RL arguments
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--offline", default=False, action=argparse.BooleanOptionalAction)  # TODO: Implement offline training
    parser.add_argument("--use_checkpoints", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--load_model", default=False, action=argparse.BooleanOptionalAction)
    # Evaluation
    parser.add_argument("--timesteps_before_training", default=25e3, type=int)
    parser.add_argument("--eval_freq", default=5e3, type=int)
    parser.add_argument("--eval_eps", default=10, type=int)
    parser.add_argument("--max_timesteps", default=5e6, type=int)
    parser.add_argument("--max_episode_steps", default=500, type=int)
    # File
    parser.add_argument("--file_name", default=None)
    arguments = parser.parse_args()

    # File name to save results
    today = date.today().strftime("%Y%m%d")
    if arguments.file_name is None:
        arguments.file_name = f"td7_agent_seed_{arguments.seed}_date_{today}"

    # Seed for reproducibility
    torch.manual_seed(arguments.seed)
    np.random.seed(arguments.seed)

    # Start the training
    main(arguments)
