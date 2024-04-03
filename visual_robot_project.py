import os
import gymnasium as gym
import panda_gym
import numpy as np
from stable_baselines3 import DDPG, A2C
from stable_baselines3.common.logger import configure
from time import sleep
from typing import Any, Dict, Tuple, Optional
from scipy.spatial.transform import Rotation as R
from panda_gym.envs.core import RobotTaskEnv
from panda_gym.envs.robots.panda import Panda
from panda_gym.envs.core import Task
from panda_gym.pybullet import PyBullet
from panda_gym.utils import angle_distance
import pybullet as p

from stable_baselines3 import HerReplayBuffer
from stable_baselines3 .common.noise import NormalActionNoise
from sb3_contrib.common.wrappers import TimeFeatureWrapper

from test_reach_task_code import CustomReachEnv

# Create the environment
env = CustomReachEnv(render_mode="human") 
# env = gym.make('PandaReach-v3', render_mode="human")
print("Environment:", env)
env = TimeFeatureWrapper(env)

model_path_1 = "./trained_models/trained_reach.zip"
model_path_2 = "./Saved_Models/2mil_reach.zip"
model_path = model_path_1

# Configure logger for TensorBoard
log_dir = "./logs/"
os.makedirs(log_dir, exist_ok=True)
if os.path.exists(model_path):

    model = DDPG.load(model_path, env=env)
    print("loaded existing model from", model_path, "\n", model)

else:

    rb_kwargs = {#'online_sampling' : True,
                'goal_selection_strategy' : 'future',
                'n_sampled_goal' : 4}

    policy_kwargs = {'net_arch' : [512, 512, 512], 
                    'n_critics' : 2}

    n_actions = env.action_space.shape[0]
    noise = NormalActionNoise(mean = np.zeros(n_actions), sigma = 0.1 * np.ones(n_actions))


    model = DDPG(policy="MultiInputPolicy", env=env, replay_buffer_class=HerReplayBuffer, verbose=1, 
                gamma = 0.95, batch_size= 2, buffer_size=10, replay_buffer_kwargs = rb_kwargs, 
                learning_rate = 1e-3, action_noise = noise, policy_kwargs = policy_kwargs)
    
# Train the agent
# model.learn(total_timesteps=5000, progress_bar=True)  # Number of timesteps to train for

# os.makedirs(os.path.dirname(model_path), exist_ok=True)
# model.save(model_path)

# Test the trained agent
# observation, info = env.reset()
# for _ in range(1000):
#     action, _states = model.predict(observation, deterministic=True)
#     observation, reward, terminated, truncated, info = env.step(action)

#     if terminated or truncated:
#         observation, info = env.reset()

# Test the trained agent
total_reward = 0
num_episodes = 10  # Number of episodes to test
for episode in range(num_episodes):
    print("Episode:",)
    observation, info = env.reset()
    episode_reward = 0
    terminated = False
    while not terminated:
        action, _states = model.predict(observation, deterministic=True)
        observation, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward

        if terminated or truncated:
            total_reward += episode_reward
            print(f"Episode {episode + 1}: Cumulative Reward = {episode_reward}")
            break

average_reward = total_reward / num_episodes
print(f"Average Cumulative Reward over {num_episodes} episodes: {average_reward}")
