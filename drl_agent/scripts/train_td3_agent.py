#!/usr/bin/env python3

import os
import rclpy
from rclpy.node import Node
from drl_agent_interfaces.srv import Step, Reset

import torch
import numpy as np

from td3_agent import TD3
from replay_buffer import ReplayBuffer


class TrainTD3(Node):
    ''' Node for training a TD3 agent

    Attributes:
    -----------
        seed: int
            Random seed number for getting consistent results
        eval_freq: int
            After how many steps to perform the evaluation
        max_ep: int
            Maximum number of steps the agent can take per episode
        eval_ep: int
            The number of episodes for evaluation
        max_timesteps: int
            Maximum number of steps to perform in the training
        expl_noise: float
            Initial exploration noise starting value, in range [expl_min, 1]
        expl_min: float
            Exploration noise after the decay, in range [0, expl_noise]
        expl_decay_steps: int
            Number of steps over which the initial exploration noise will decay over
        batch_size: int
            Size of the batch to sample from the replay-buffer for training
        discount: float
            Discount factor for calculating the discounted return, should be close to 1
        tau: float
            Soft target update variable, should be close to 0
        policy_noise: float
            Added noise for exploration
        noise_clip: float
            Maximum clamping values of the noise
        policy_freq: int
            Actor network update frequency
        buffer_size: int
            Maximum size of the replay-buffer
        file_name: str
            File name to save the policy
        save_model: bool
            Whether to save the model or not
        load_model: bool
            Whether to load a stored model
        random_near_obstacle: bool
            Whether to take random actions near obstacles or not
        environment_dim: int
            State size from the environment
        robot_dim: int
            Additional states for the agent: heading, dist_from_goal, last actions
        state_dim: int
            Overall state size: environment_dim + robot_dim
        action_dim: int
            Number of available actions for the agent
        max_action: float
            Max. allowable value for the actions tobe taken, physical limits
    '''

    def __init__(self):
        # Node: train_td3_node
        super().__init__('train_td3_node')

        # Get the device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

        # Setup dirs for saving models, logs and results
        drl_agent_pkg_dir_env = 'DRL_AGENT_PACKAGE_PATH'     # echo 'export DRL_AGENT_PACKAGE_PATH="/<path/to>/src/drl_agent"' >> ~/.bashrc
        drl_agent_pkg_dir = os.getenv(drl_agent_pkg_dir_env)

        if drl_agent_pkg_dir is None:
            self.get_logger().error(f'Environment variable: {drl_agent_pkg_dir_env} is not set.')

        self.pytorch_models_dir = os.path.join(drl_agent_pkg_dir, 'scripts', 'pytorch_models')
        self.final_models_dir = os.path.join(drl_agent_pkg_dir, 'scripts', 'final_models')
        self.results_dir = os.path.join(drl_agent_pkg_dir, 'scripts', 'results')
        self.log_dir = os.path.join(drl_agent_pkg_dir, 'scripts', 'logs')

        # Create the directories, if they don't exist
        os.makedirs(self.pytorch_models_dir, exist_ok=True)
        os.makedirs(self.final_models_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

        # Log intro info 
        self.get_logger().info('+===--------------------------------------===+')
        self.get_logger().info(f'| > Device: {self.device}')
        self.get_logger().info('+===--------------------------------------===+')
        self.get_logger().info('| > Results will be saved in:')
        self.get_logger().info(f'|\t> {self.pytorch_models_dir}')
        self.get_logger().info(f'|\t> {self.final_models_dir}')
        self.get_logger().info(f'|\t> {self.results_dir}')
        self.get_logger().info(f'|\t> {self.log_dir}')
        self.get_logger().info('+===--------------------------------------===+\n')


        # Set the parameters for the training
        self.seed = 40  
        self.eval_freq = 5e3 
        self.max_ep = 500 
        self.eval_ep = 10  
        self.max_timesteps = 5e6 
        self.expl_noise = 1  
        self.expl_decay_steps = 500000 
        self.expl_min = 0.1  
        self.batch_size = 40  
        self.discount = 0.99999  
        self.tau = 0.005  
        self.policy_noise = 0.2  
        self.noise_clip = 0.5  
        self.policy_freq = 2  
        self.buffer_size = 1e6  
        self.file_name = 'td3_agent'  
        self.save_model = True  
        self.load_model = False
        self.random_near_obstacle = True 

        # Specify the training environment
        self.environment_dim = 20
        self.robot_dim = 4
        self.state_dim = self.environment_dim + self.robot_dim
        self.action_dim = 2
        self.max_action = 1

        # Seed
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        
        # Create the network
        self.network = TD3(self.state_dim, self.action_dim, self.max_action, self.device, self.log_dir)
        # Create a replay buffer
        self.replay_buffer = ReplayBuffer(self.buffer_size, self.seed)
        if self.load_model:
            try:
                self.network.load(self.file_name, self.pytorch_models_dir)
            except:
                self.get_logger().warning('!!! The stored model parameters couldn\'t be loaded, initializing training with random parameters...')

        # Create service client and wait services to be online: step and reset tobecome online
        self.reset_client = self.create_client(Reset, 'reset')
        self.step_client = self.create_client(Step, 'step')
        while not self.reset_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service /reset not available, waiting again...')
        while not self.step_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service /step not available, waiting again...')
        
        # Flag to indicate training is done
        self.training_is_done = False
 
    def reset_environment(self):
        ''' Resets the environment to its initial state using /reset service.

        Parameters
        ----------
        None

        Returns
        -------
        state : list
            The initial state of the environment after reset.
        '''
        req = Reset.Request()
        future = self.reset_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        return future.result().state
    
    def step_environment(self, action):
        ''' Takes a step in the environment with the given action.

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
        '''
        req = Step.Request()
        req.action = action
        future = self.step_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        response = future.result()
        return response.state, response.reward, response.done, response.target
    
    def evaluate(self, epoch, eval_episodes=10):
        ''' Evaluates the performance of the agent over a given number of episodes.

        Runs the agent in the environment for a specified number of evaluation episodes, 
        calculates the average reward and average number of collisions, and logs the evaluation results.

        Parameters
        ----------
        epoch : int
            The current epoch number of the training process.
        eval_episodes : int, optional
            The number of episodes to run for evaluation.

        Returns
        -------
        avg_reward : float
            The average reward over the evaluation episodes.
        '''

        self.get_logger().info(f'{" EVALUATION AT EPOCH " + str(epoch) + " ":-^55}')
        avg_reward = 0.0
        col = 0
        for _ in range(eval_episodes):
            count = 0
            state = self.reset_environment()
            done = False
            while not done and count < 501:
                action = self.network.get_action(np.array(state))
                a_in = np.array([(action[0] + 1)/2, action[1]], dtype=np.float64).tolist()  # Ensure it matches the service type
                state, reward, done, _ = self.step_environment(a_in)
                avg_reward += reward
                count += 1
                if reward < -90:
                    col += 1
        avg_reward /= eval_episodes
        avg_col = col / eval_episodes

        border = '+===-------------------------------------------===+'
        self.get_logger().info(border)
        self.get_logger().info(f'| > Evaluation episodes: {eval_episodes:<23} |')
        self.get_logger().info(f'| > Epoch: {epoch:<37} |')
        self.get_logger().info(f'| > Average reward: {avg_reward:<28.4f} |')
        self.get_logger().info(f'| > Average col: {avg_col:<31.4f} |')
        self.get_logger().info(border + '\n')

        return avg_reward
    
    def train(self):
        ''' Interface function for training the agent.

        This function initializes the training process, handles the main training loop, 
        manages exploration noise, and evaluates and saves the agent's performance periodically. 

        Parameters
        ----------
        None

        Returns
        -------
        None
        '''

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
        while timestep < self.max_timesteps:

            # On termination of an episode
            if done:
                if timestep != 0:
                    self.network.train(
                        self.replay_buffer,
                        episode_timesteps,
                        self.batch_size,
                        self.discount,
                        self.tau,
                        self.policy_noise,
                        self.noise_clip,
                        self.policy_freq,
                    )

                if timesteps_since_eval >= self.eval_freq:
                    self.get_logger().info(f'{" Validating ":=^55}')
                    timesteps_since_eval %= self.eval_freq
                    evaluations.append(
                        self.evaluate(epoch=epoch, eval_episodes=self.eval_ep)
                    )
                    self.network.save(self.file_name, directory=self.pytorch_models_dir)
                    np.save(os.path.join(self.results_dir, f'{self.file_name}.npy'), evaluations)
                    epoch += 1

                state = self.reset_environment()
                done = False

                episode_reward = 0
                episode_timesteps = 0
                episode_num += 1

            # add some exploration noise
            if self.expl_noise > self.expl_min:
                self.expl_noise = self.expl_noise - ((1 - self.expl_min) / self.expl_decay_steps)

            action = self.network.get_action(np.array(state))
            action = (action + np.random.normal(0, self.expl_noise, size=self.action_dim)).clip(-self.max_action, self.max_action)

            # If the robot is facing an obstacle, randomly force it to take a consistent random action to increase exploration
            # Training can also be performed without it
            if self.random_near_obstacle:
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

            # Update action to fall in range [0,1] for linear velocity and [-1,1] for angular velocity
            a_in = np.array([(action[0] + 1) / 2, action[1]], dtype=np.float64).tolist() # Ensure it matches the service type
            next_state, reward, done, target = self.step_environment(a_in)
            done_bool = 0 if episode_timesteps + 1 == self.max_ep else int(done)
            done = 1 if episode_timesteps + 1 == self.max_ep else int(done)
            episode_reward += reward

            # Save the tuple in the replay buffer
            self.replay_buffer.add(state, action, reward, done_bool, next_state)

            # Update the counters
            state = next_state
            episode_timesteps += 1
            timestep += 1
            timesteps_since_eval += 1

        # After the training is done, evaluate the network and save it
        evaluations.append(self.evaluate(epoch=epoch, eval_episodes=self.eval_ep))
        if self.save_model:
            self.network.save(self.file_name, directory=self.final_models_dir)
        np.save(os.path.join(self.results_dir, f'{self.file_name}.npy'), evaluations)

        # Indicate training is done
        self.training_is_done = True

def main(args=None):
    # Initialize ROS2 communication
    rclpy.init(args=args)
    # Initialize the train_td3_node object
    train_td3 = TrainTD3()

    # Invike training
    train_td3.train()

    try:
        while rclpy.ok():
            rclpy.spin_once(train_td3)
            if train_td3.training_is_done:
                break
    finally:
        train_td3.get_logger().info(f'\n\t{" Training is Done ":=^50}\n')
        train_td3.get_logger().info('rclpy, shutingdown...')
        train_td3.destroy_node()
        rclpy.shutdown()

if __name__=='__main__':
    main()

