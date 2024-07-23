#!/usr/bin/env python3

import os
import argparse
from datetime import date
import rclpy
from rclpy.node import Node
from drl_agent_interfaces.srv import Step, Reset, SampleActionSpace, Seed, GetDimensions
import torch
import numpy as np
from td3_agent import TD3
from replay_buffer import ReplayBuffer


class TrainTD3(Node):
    """Node for training a TD3 agent"""

    def __init__(self, args):
        # Node: train_td3_node
        super().__init__("train_td3_node")
        self.args = args

        # Setup dirs for saving models, logs and results
        drl_agent_pkg_dir_env = "DRL_AGENT_PACKAGE_PATH"  # echo 'export DRL_AGENT_PACKAGE_PATH=~/drl_agent_ws/src/drl_agent' >> ~/.bashrc
        drl_agent_pkg_dir = os.getenv(drl_agent_pkg_dir_env)
        if drl_agent_pkg_dir is None:
            self.get_logger().error(
                f"Environment variable: {drl_agent_pkg_dir_env} is not set."
            )

        self.pytorch_models_dir = os.path.join(
            drl_agent_pkg_dir, "scripts", "pytorch_models"
        )
        self.final_models_dir = os.path.join(
            drl_agent_pkg_dir, "scripts", "final_models"
        )
        self.results_dir = os.path.join(drl_agent_pkg_dir, "scripts", "results")
        self.log_dir = os.path.join(drl_agent_pkg_dir, "scripts", "logs")
        self.file_name = self.args.file_name

        # Create the directories, if they don't exist
        os.makedirs(self.pytorch_models_dir, exist_ok=True)
        os.makedirs(self.final_models_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

        # Create service clients
        self.reset_client = self.create_client(Reset, "reset")
        self.step_client = self.create_client(Step, "step")
        self.seed_client = self.create_client(Seed, "seed")
        self.dimensions_client = self.create_client(GetDimensions, "get_dimensions")
        self.actio_space_sample_client = self.create_client(
            SampleActionSpace, "action_space_sample"
        )

        # Set seed of for environment
        self.set_env_seed(self.args.seed)
        # Create a replay buffer
        self.replay_buffer = ReplayBuffer(self.args.buffer_size, self.args.seed)
        # Get the dimensions of the environment
        self.state_dim, self.action_dim, self.max_action = self.get_dimensions()
        # Create the network
        self.network = TD3(
            self.state_dim, self.action_dim, self.max_action, self.log_dir
        )
        if self.args.load_model:
            try:
                self.network.load(self.pytorch_models_dir, self.file_name)
                self.get_logger().info("Models loaded")
            except Exception as e:
                self.get_logger().warning(
                    f"!!! The stored model parameters couldn't be loaded: {str(e)}"
                )
                self.get_logger().warning("!!! Training from scratch...")

        # Flag to indicate training is done
        self.done_training = False

        # Log intro info
        self.border = "+" + "-" * 80 + "+"
        self.get_logger().info(self.border)
        self.get_logger().info("Hyperparameters:")
        self.get_logger().info(f"{self.args}")
        self.get_logger().info(self.border)
        self.get_logger().info(f"| File name:{self.file_name} | Seed: {self.args.seed}")
        self.get_logger().info(self.border)
        self.get_logger().info("| Results will be saved in:")
        self.get_logger().info(f"|  {self.pytorch_models_dir}")
        self.get_logger().info(f"|  {self.final_models_dir}")
        self.get_logger().info(f"|  {self.results_dir}")
        self.get_logger().info(f"|  {self.log_dir}")
        self.get_logger().info("| Environment")
        self.get_logger().info(self.border)
        self.get_logger().info(self.border)
        self.get_logger().info(f"| State Dim: {self.state_dim}")
        self.get_logger().info(f"| Action Dim: {self.action_dim}")
        self.get_logger().info(f"| Max Action: {self.max_action}")
        self.get_logger().info(self.border)

    def set_env_seed(self, seed):
        """Set the seed of the environment"""
        request = Seed.Request()
        request.seed = seed
        while not self.seed_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Service /seed not available, waiting again...")
        try:
            future = self.seed_client.call_async(request)
            rclpy.spin_until_future_complete(self, future)
        except Exception as e:
            self.get_logger().error(f"Service call /seed failed: {e}")
        self.get_logger().info(f"Environment seed set to: {seed}, success: {future.result().success}")

    def get_dimensions(self):
        """Get the dimensions of the environment"""
        request = GetDimensions.Request()
        while not self.dimensions_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info(
                "Service /get_dimensions not available, waiting again..."
            )
        try:
            future = self.dimensions_client.call_async(request)
            rclpy.spin_until_future_complete(self, future)
        except Exception as e:
            self.get_logger().error(f"Service call /get_dimensions failed: {e}")
        response = future.result()
        return response.state_dim, response.action_dim, response.max_action

    def sample_action_space(self):
        """Sample an action from the action space"""
        request = SampleActionSpace.Request()
        while not self.actio_space_sample_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info(
                "Service /action_space_sample not available, waiting again..."
            )
        try:
            future = self.actio_space_sample_client.call_async(request)
            rclpy.spin_until_future_complete(self, future)
        except Exception as e:
            self.get_logger().error(f"Service call /action_space_sample failed: {e}")
        return future.result().action

    def reset_environment(self):
        """Resets the environment and returns the initial state"""
        req = Reset.Request()
        while not self.reset_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Service /reset not available, waiting again...")
        try:
            future = self.reset_client.call_async(req)
            rclpy.spin_until_future_complete(self, future)
        except Exception as e:
            self.get_logger().error(f"Service call /reset failed: {e}")
        return future.result().state

    def step_environment(self, action):
        """Takes a step in the environment with the given action.

        Sends an action request to the environment, waits for the step to complete, and returns
        the resulting state, reward, done flag, and target flag.
        """
        req = Step.Request()
        req.action = action
        while not self.step_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Service /step not available, waiting again...")
        try:
            future = self.step_client.call_async(req)
            rclpy.spin_until_future_complete(self, future)
        except Exception as e:
            self.get_logger().error(f"Service call /step failed: {e}")
        response = future.result()
        return response.state, response.reward, response.done, response.target

    def save_models(self, directory, file_name):
        """Save the models at the given step"""
        self.network.save(directory, file_name)
        self.get_logger().info("Models updated")

    def evaluate(self, epoch, eval_episodes=10):
        """Evaluates the performance of the agent over a given number of episodes.

        Runs the agent in the environment for a specified number of evaluation episodes,
        calculates the average reward and average number of collision_countlisions, and logs the evaluation results.
        """
        self.get_logger().info(f'{" EVALUATION AT EPOCH " + str(epoch) + " ":-^55}')
        avg_reward = 0.0
        collision_count = 0
        for _ in range(eval_episodes):
            count = 0
            state, done = self.reset_environment(), False
            while not done and count < 501:
                action = self.network.get_action(np.array(state))
                action_cmd = np.array(
                    [(action[0] + 1) / 2, action[1]], dtype=np.float32
                )
                state, reward, done, _ = self.step_environment(action_cmd.tolist())
                avg_reward += reward
                count += 1
                if reward < -90:
                    collision_count += 1
        avg_reward /= eval_episodes
        avg_col = collision_count / eval_episodes

        self.get_logger().info(self.border)
        self.get_logger().info(f"| > Evaluation episodes: {eval_episodes}")
        self.get_logger().info(f"| > Average reward: {avg_reward}")
        self.get_logger().info(f"| > Average collision: {avg_col}")
        self.get_logger().info(self.border + "\n")

        return avg_reward

    def train(self):
        """Interface function for training the agent"""

        # Initialize training variables
        timestep = 0
        timesteps_since_eval = 0
        episode_num = 0
        done = True
        epoch = 1
        count_rand_actions = 0
        random_action = []
        # Create evaluation data store
        evaluations = []

        # Begin the training loop
        while timestep < self.args.max_timesteps:
            if done:  # On termination of an episode
                if timestep != 0:
                    self.network.train(
                        self.replay_buffer,
                        episode_timesteps,
                        self.args.batch_size,
                        self.args.discount,
                        self.args.tau,
                        self.args.policy_noise,
                        self.args.noise_clip,
                        self.args.policy_freq,
                    )

                if timesteps_since_eval >= self.args.eval_freq:
                    self.get_logger().info(f'{" Validating ":=^55}')
                    timesteps_since_eval %= self.args.eval_freq
                    evaluations.append(
                        self.evaluate(epoch=epoch, eval_episodes=self.args.eval_ep)
                    )
                    self.save_models(self.pytorch_models_dir, self.file_name)
                    np.save(
                        os.path.join(self.results_dir, f"{self.file_name}.npy"),
                        evaluations,
                    )
                    epoch += 1

                state, done = self.reset_environment(), False
                episode_reward = 0
                episode_timesteps = 0
                episode_num += 1

            # add some exploration noise
            if self.args.expl_noise > self.args.expl_min:
                self.args.expl_noise = self.args.expl_noise - (
                    (1 - self.args.expl_min) / self.args.expl_decay_steps
                )

            action = self.network.get_action(np.array(state))
            action = (action + np.random.normal(0, self.args.expl_noise, size=self.action_dim)).clip(-self.max_action, self.max_action)

            # If the robot is facing an obstacle, randomly force it to take a consistent random action to increase exploration
            # Training can also be performed without it
            if self.args.random_near_obstacle:
                if (
                    np.random.uniform(0, 1) > 0.85
                    and min(state[4:-8]) < 0.6
                    and count_rand_actions < 1
                ):
                    count_rand_actions = np.random.randint(8, 15)
                    random_action = np.random.uniform(-1, 1, 2)

                if count_rand_actions > 0:
                    count_rand_actions -= 1
                    action = random_action
                    action[0] = -1

            # Update action so that liner vel. falls in range [0,1]
            action_cmd = np.array([(action[0] + 1) / 2, action[1]], dtype=np.float32)
            next_state, reward, done, _ = self.step_environment(action_cmd.tolist())
            done_bool = 0 if episode_timesteps + 1 == self.args.max_ep else int(done)
            done = 1 if episode_timesteps + 1 == self.args.max_ep else int(done)
            episode_reward += reward
            # Save the tuple in the replay buffer
            self.replay_buffer.add(state, action, reward, done_bool, next_state)

            # Update counters
            state = next_state
            episode_timesteps += 1
            timestep += 1
            timesteps_since_eval += 1

        # After the training is done, evaluate the network and save it
        evaluations.append(self.evaluate(epoch=epoch, eval_episodes=self.args.eval_ep))
        if self.args.save_model:
            self.network.save(self.final_models_dir, self.file_name)
        np.save(os.path.join(self.results_dir, f"{self.file_name}.npy"), evaluations)
        # Indicate training is done
        self.done_training = True


def main(arguments, args=None):
    # Initialize ROS2 communication
    rclpy.init(args=args)
    # Initialize the train_td3_node object
    train_td3 = TrainTD3(arguments)

    # Invike training
    train_td3.train()

    try:
        while rclpy.ok() and not train_td3.done_training:
            rclpy.spin_once(train_td3)
    except KeyboardInterrupt:
        train_td3.get_logger().info("Keyboard interrupt detected, stopping training...")
        if arguments.save_model:
            train_td3.network.save(train_td3.final_models_dir, arguments.file_name)
            train_td3.get_logger().info(
                f"Final models saved in: {train_td3.final_models_dir}"
            )
    finally:
        train_td3.get_logger().info(f'\n\t{" Training done ":-^50}\n')
        train_td3.get_logger().info("rclpy, shutingdown...")
        train_td3.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    # Set the parameters for the training
    parser = argparse.ArgumentParser(description="Train TD3 agent")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval_freq", type=int, default=5e3)
    parser.add_argument("--max_ep", type=int, default=500)
    parser.add_argument("--eval_ep", type=int, default=10)
    parser.add_argument("--max_timesteps", type=int, default=5e6)
    parser.add_argument("--expl_noise", type=float, default=1.0)
    parser.add_argument("--expl_decay_steps", type=int, default=500000)
    parser.add_argument("--expl_min", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=40)
    parser.add_argument("--discount", type=float, default=0.99999)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--policy_noise", type=float, default=0.2)
    parser.add_argument("--noise_clip", type=float, default=0.5)
    parser.add_argument("--policy_freq", type=int, default=2)
    parser.add_argument("--buffer_size", type=int, default=1e6)
    parser.add_argument(
        "--save_model", default=True, action=argparse.BooleanOptionalAction
    )
    parser.add_argument(
        "--load_model", default=False, action=argparse.BooleanOptionalAction
    )
    parser.add_argument(
        "--random_near_obstacle", default=True, action=argparse.BooleanOptionalAction
    )
    parser.add_argument("--file_name", default=None)
    arguments = parser.parse_args()

    if arguments.file_name is None:
        today = date.today().strftime("%Y%m%d")
        arguments.file_name = f"td3_agent_seed_{arguments.seed}_{today}"

    # Seed for reproducibility
    torch.manual_seed(arguments.seed)
    np.random.seed(arguments.seed)

    main(arguments)
