#!/usr/bin/env python3

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import rclpy
from rclpy.node import Node
from drl_agent_interfaces.srv import Reset, Step

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()

        self.layer_1 = nn.Linear(state_dim, 800)
        self.layer_2 = nn.Linear(800, 600)
        self.layer_3 = nn.Linear(600, action_dim)
        self.tanh = nn.Tanh()

    def forward(self, s):
        s = F.relu(self.layer_1(s))
        s = F.relu(self.layer_2(s))
        a = self.tanh(self.layer_3(s))
        return a


class TD3(object):
    def __init__(self, state_dim, action_dim):
        # Initialize the Actor network
        self.actor = Actor(state_dim, action_dim).to(device)

    def get_action(self, state):
        """Get action from the actor"""
        state = torch.Tensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def load(self, filename, directory):
        """Load network parameters"""
        self.actor.load_state_dict(
            torch.load("%s/%s_actor.pth" % (directory, filename))
        )


class TestTD3(Node):
    """Node for testing td3 agent

    Attributes:
    -----------
        seed: int
            Random seed number
        max_ep: int
            Maximum number of steps per episode
        file_name: str
            Name of the file to load the policy from

        NOTE: the floowing should match with the ones defined in train_td3_agent.py
        environment_dim: int
            State size from the environment
        robot_dim: int
            Additional states for the agent: heading, dist_from_goal, last actions
        state_dim: int
            Overall state size: environment_dim + robot_dim
        action_dim: int
            Number of available actions for the agent
    """

    def __init__(self):
        super().__init__("test_td3_node")

        # Get the saved model directory
        drl_agent_pkg_dir_env = "DRL_AGENT_PACKAGE_PATH"  # echo 'export DRL_AGENT_PACKAGE_PATH=~/drl_agent_ws/src/drl_agent' >> ~/.bashrc
        drl_agent_pkg_dir = os.getenv(drl_agent_pkg_dir_env)

        if drl_agent_pkg_dir is None:
            self.get_logger().error(f"Environment variable: {drl_agent_pkg_dir_env} is not set.")
        self.pytorch_models_dir = os.path.join(drl_agent_pkg_dir, "scripts", "pytorch_models")

        # Set the parameters for the implementation
        self.seed = 40
        self.max_ep = 500
        self.file_name = "td3_agent"

        # Create the testing environment
        self.environment_dim = 20
        self.robot_dim = 4
        self.state_dim = self.environment_dim + self.robot_dim
        self.action_dim = 2

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        # Create the network
        self.network = TD3(self.state_dim, self.action_dim)
        try:
            self.network.load(self.file_name, self.pytorch_models_dir)
        except Exception as e:
            self.get_logger().error("Could not load the stored model parameters :(")

        # Create service client and wait for services to be online: step and reset tobecome online
        self.reset_client = self.create_client(Reset, "reset")
        self.step_client = self.create_client(Step, "step")
        while not self.reset_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Service /reset not available, waiting again...")
        while not self.step_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Service /step not available, waiting again...")

        # Flag to indicate testing is done
        # self.testing_is_done = False

    def reset_environment(self):
        """Resets the environment to its initial state using /reset service.

        Parameters
        ----------
        None

        Returns
        -------
        state : list
            The initial state of the environment after reset.
        """
        req = Reset.Request()
        future = self.reset_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        return future.result().state

    def step_environment(self, action):
        """Takes a step in the environment with the given action.

        Sends an action request to the environment, waits for the step to complete, and returns
        the resulting state, reward, done flag, and target flag.

        Parameters
        ----------
        action : list
            The action to be performed by the agent.

        Returns
        -------
        tuple
            A tuple containing:
            state : list
                The next state of the environment.
            reward : float
                The reward received for the action.
            done : bool
                The done flag indicating if the episode has ended.
            target : bool
                The target flag indicating if the goal was reached.
        """
        req = Step.Request()
        req.action = action
        future = self.step_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        response = future.result()
        return response.state, response.reward, response.done, response.target

    def test(self):
        """Runs a continuous testing loop for the agent.

        Resets the environment, then continuously takes actions according to the agent's policy,
        stepping through the environment and resetting when episodes terminate.
        """
        done = False
        episode_timesteps = 0
        state = self.reset_environment()

        # Begin the testing loop
        while True:
            action = self.network.get_action(np.array(state))

            # Update action to fall in range [0,1] for linear velocity and [-1,1] for angular velocity
            a_in = np.array([(action[0] + 1) / 2, action[1]], dtype=np.float64).tolist()  # Ensure it matches the service type
            next_state, reward, done, target = self.step_environment(a_in)
            done = 1 if episode_timesteps + 1 == self.max_ep else int(done)

            # On termination of episode
            if done:
                state = self.reset_environment()
                done = False
                episode_timesteps = 0
            else:
                state = next_state
                episode_timesteps += 1

        # Indicate that testing is done
        # self.testing_is_done = True


def main(args=None):
    # Initialize ROS2 communication
    rclpy.init(args=args)
    # Initialize the train_td3_node object
    test_td3 = TestTD3()

    # Start testing
    test_td3.test()

    try:
        while rclpy.ok():
            rclpy.spin_once(test_td3)
            # if test_td3.testing_is_done:
            #     break
    finally:
        test_td3.get_logger().info(f'\n\t{" Testing is Done ":=^50}\n')
        test_td3.get_logger().info("rclpy, shutingdown...")
        test_td3.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
