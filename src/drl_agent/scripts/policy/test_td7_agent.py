#!/usr/bin/env python3

import os
import sys
import time
import torch
import numpy as np

import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import QoSProfile, ReliabilityPolicy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup

from nav_msgs.msg import Odometry
from td7_agent import Agent
from environment_interface import EnvInterface
from file_manager import load_yaml, save_yaml, save_json


class TestTD7(EnvInterface):
    def __init__(self):
        super().__init__("test_td7_agent")

        """************************************************
		** Test mode
		************************************************"""
        # Determine if the test is to be run in a random or determnistic way
        self.declare_parameter("test_mode", "random_test")
        self.test_mode = (
            self.get_parameter("test_mode").get_parameter_value().string_value.lower()
        )
        if not self.test_mode in ["test", "random_test"]:
            raise NotImplementedError
        # Test run mode
        self.random_train_mode = self.test_mode == "random_test"
        self.get_logger().info(f"Test run mode: {self.test_mode}")

        """************************************************
		** Test config
		************************************************"""
        # Base directory for loading models and for saving trajectories
        drl_agent_src_path_env = "DRL_AGENT_SRC_PATH"
        drl_agent_src_path = os.getenv(drl_agent_src_path_env)
        if drl_agent_src_path is None:
            self.get_logger().error(
                f"Environment variable: {drl_agent_src_path_env} is not set"
            )
            sys.exit(-1)
        drl_agent_pkg_path = os.path.join(drl_agent_src_path, "drl_agent")

        test_config_file_path = os.path.join(
            drl_agent_pkg_path, "config", "test_config.yaml"
        )
        self.pytorch_models_dir = os.path.join(
            drl_agent_pkg_path, "temp", "pytorch_models"
        )
        self.hyperparameters_path = os.path.join(
            drl_agent_pkg_path, "config", "hyperparameters.yaml"
        )

        # Trajectories will be save in
        self.test_metric_dir = os.path.join(drl_agent_pkg_path, "test_runs")
        os.makedirs(self.test_metric_dir, exist_ok=True)

        # Load test config file
        try:
            self.test_config = load_yaml(test_config_file_path)["test_settings"]
        except Exception as e:
            self.get_logger().info(f"Unable to load config file: {e}")
        self.seed = self.test_config["seed"]
        save_date = self.test_config["save_date"]
        base_file_name = self.test_config["base_file_name"]
        self.file_name = f"{base_file_name}_seed_{self.seed}_{save_date}"
        self.use_checkpoints = self.test_config["use_checkpoints"]
        self.max_episode_steps = self.test_config["max_episode_steps"]

        # Set seed
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        self.set_env_seed(self.seed)

        # Initialize the agent
        state_dim, action_dim, max_action = self.get_dimensions()
        try:
            hp = load_yaml(self.hyperparameters_path)["hyperparameters"]
        except Exception as e:
            self.get_logger().error(f"Unable to load hyperprameters file: {e}")
            sys.exit(-1)
        self.rl_agent = Agent(
            state_dim=state_dim, action_dim=action_dim, max_action=max_action, hp=hp
        )
        try:
            self.rl_agent.load(self.pytorch_models_dir, self.file_name)
            self.get_logger().info(f'{"Model parameters loaded successfuly":-^50}')
        except Exception as e:
            self.get_logger().error(f'{"Could not load trained models :(":-^50}')
            sys.exit(-1)

        # Callback groups for handling sensors and services in parallel
        self.odom_callback_group = MutuallyExclusiveCallbackGroup()
        # Initialize subscribers
        qos_profile = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        self.odom = self.create_subscription(
            Odometry,
            "odom",
            self.odom_callback,
            qos_profile,
            callback_group=self.odom_callback_group,
        )
        self.odom
        self.last_odom = None

        # Initialize metrics
        self.all_episode_times = []
        self.all_episode_distances = []
        self.all_trajectories = []
        self.target_reached_counter = 0.0
        self.num_episodes_counter = 0.0

        self.test()

    def odom_callback(self, od_data):
        """Updates the latest odometry data"""
        self.last_odom = od_data

    def test(self):
        """Run continious testing loop"""
        # Initialization
        done = False
        current_trajectory = []
        episode_timesteps = 0
        state = self.reset()

        episode_start_time = time.time()
        while True:
            action = self.rl_agent.select_action(
                np.array(state), self.use_checkpoints, use_exploration=False
            )
            next_state, reward, done, target = self.step(action)
            done = 1 if episode_timesteps + 1 == self.max_episode_steps else int(done)

            # Record current position
            if not self.last_odom is None:
                x = self.last_odom.pose.pose.position.x
                y = self.last_odom.pose.pose.position.y
                current_trajectory.append({"x": x, "y": y})

            # On termination of episode
            if done:
                # Save recorded trajectory and reset current trajectory
                self.all_trajectories.append(current_trajectory)
                self.num_episodes_counter += 1
                # Calculate metrics
                if target:
                    self.all_episode_distances.append(
                        self.calculate_distance(current_trajectory)
                    )
                    self.all_episode_times.append(time.time() - episode_start_time)
                    self.target_reached_counter += 1
                # Save metrics
                self.save_test_metrics()

                # Reset
                state = self.reset()
                done = False
                episode_timesteps = 0
                current_trajectory = []
                episode_start_time = time.time()
            else:
                state = next_state
                episode_timesteps += 1

    def calculate_distance(self, traj):
        """Calculates the total distance traveled in a trajectory"""
        distance = 0.0
        for i in range(1, len(traj)):
            x1, y1 = traj[i - 1]["x"], traj[i - 1]["y"]
            x2, y2 = traj[i]["x"], traj[i]["y"]
            distance += np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        return distance

    def save_test_metrics(self):
        # Specify file names
        traj_filename = os.path.join(self.test_metric_dir, "ours_test_run.json")
        metrics_filename = os.path.join(self.test_metric_dir, "ours_test_run.yaml")

        # Perform test_metrics computation
        average_time = round(float(np.mean(self.all_episode_times)), 4)
        average_distance = round(float(np.mean(self.all_episode_distances)), 4)
        success_rate = round(self.target_reached_counter / self.num_episodes_counter, 4)
        collision_rate = 1 - success_rate

        data = {
            "test_metrics": {
                "average_time": average_time,
                "average_distance": average_distance,
                "success_rate": success_rate,
                "collision_rate": collision_rate,
            }
        }

        # Save trajectory
        try:
            save_json(traj_filename, self.all_trajectories)
        except Exception as e:
            self.get_logger().error(f"Unable to save trajectories: {e}")
            sys.exit(-1)
        self.get_logger().info(f"Saved trajectory data to: {traj_filename}")
        # Save metrics
        try:
            save_yaml(metrics_filename, data)
        except Exception as e:
            self.get_logger().error(f"Unable to save metrics: {e}")
            sys.exit(-1)
        self.get_logger().info(f"Saved metrics data to: {metrics_filename}")


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


if __name__ == "__main__":
    main()
