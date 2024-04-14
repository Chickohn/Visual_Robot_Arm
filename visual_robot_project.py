import os
import time
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
from stable_baselines3.common.noise import NormalActionNoise
from sb3_contrib.common.wrappers import TimeFeatureWrapper
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy

from custom_reach import CustomReachEnv

# Create the environment
# env = CustomReachEnv(render_mode="human")

### Uncomment the following to switch to the built-in panda reach environment.
# env = gym.make('PandaReachDense-v3', render_mode="human")
env = gym.make('PandaStack-v3', render_mode="human")
# env = gym.make('PandaCustom-v3', render_mode='human')
# print(env.action_space)
# observation, info = env.reset()

# import re

# def parse_action(input_command):
#     # Default action configuration
#     action = [0, 0, 0, 0]

#     # Regex to parse commands like 'r5c'
#     match = re.match(r"([a-z]+)(\d+)?(c)?", input_command)
#     if match:
#         command, value, close_gripper = match.groups()
#         value = float(value) if value else 5  # default to 1 if no value specified

#         # Convert value into appropriate scale (divide by 10 for now as per your example)
#         value /= 10

#         # Define action mappings
#         if command == "d":
#             action[2] = -value
#         elif command == "u":
#             action[2] = value
#         elif command == "o":
#             action[3] = value
#         elif command == "c" and not close_gripper:  # Handle 'c' command only if it's not a gripper command
#             action[3] = -value
#         elif command == "b":
#             action[0] = -value
#         elif command == "f":
#             action[0] = value
#         elif command == "l":
#             action[1] = value
#         elif command == "r":
#             action[1] = -value
#         elif command == "lift":
#             action[2] = value
#             action[3] = -value
        
#         # Check if there's a gripper close command ('c' at the end)
#         if close_gripper:
#             action[3] = -0.5

#     return action

# while True:
#     action = input("\n")
    
#     if action == "stop":
#         env.close()
#         break
#     action = parse_action(action)
#     print(action)
#     observation, reward, terminated, truncated, info = env.step(action)
#     print(env.get_local_grip_position())
#     print(info)
#     print(reward)
    
#     if terminated or truncated:
#         observation, info = env.reset()

# exit()

model_path_1 = "./trained_models/trained_reach.zip"
model_path_2 = "./Saved_Models/2mil_reach.zip"
model_path = model_path_1

# Configure logger for TensorBoard
log_dir = "./logs/"
os.makedirs(log_dir, exist_ok=True)

env = make_vec_env(lambda: env, n_envs=1)

n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

# Create the DDPG model
model = DDPG("MultiInputPolicy", 
             env, 
             action_noise=action_noise, 
             replay_buffer_class=HerReplayBuffer, 
             verbose=1, 
             batch_size= 128, 
             buffer_size=1_000_000, 
             learning_rate=0.001, 
             tensorboard_log=log_dir)

# model = DDPG.load(("ddpg_model_05_00.zip"), env=env)
# model.load_replay_buffer("ddpg_buffer_1712786288.0423434")

model.learn(total_timesteps=50_000)

# from datetime import datetime

# # Get the current time
# now = datetime.now()

# # Format the time to include only hours and minutes
# current_time = now.strftime("%H:%M")
# current_time = current_time[:2]+'_'+current_time[-2:]
# print(current_time)

# model.save("ddpg_model_"+current_time)
# model.save_replay_buffer("ddpg_buffer_"+current_time)

# model = DDPG.load("ddpg_model_05_00.zip", env=env)

mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=5)

print(f"Mean reward: {mean_reward}, Std reward: {std_reward}")

# Test the model
observation = env.reset()
for i in range(1000):
    action, _states = model.predict(observation, deterministic=True)
    # try:
    observation, reward, done, info = env.step(action)
    if done:
        observation = env.reset()
    # except:
    #     pass
        # print(env.step(action))

env.close()


# if os.path.exists(model_path):

#     model = DDPG.load(model_path, env=env)
#     print("loaded existing model from", model_path, "\n", model)

# else:
#     env = TimeFeatureWrapper(env)

#     rb_kwargs = {#'online_sampling' : True,
#                 'goal_selection_strategy' : 'future',
#                 'n_sampled_goal' : 4}

#     policy_kwargs = {'net_arch' : [512, 512, 512], 
#                     'n_critics' : 2}

#     n_actions = env.action_space.shape[0]
#     noise = NormalActionNoise(mean = np.zeros(n_actions), sigma = 0.1 * np.ones(n_actions))


#     model = DDPG(policy="MultiInputPolicy", env=env, replay_buffer_class=HerReplayBuffer, verbose=1, 
#                 gamma = 0.95, batch_size= 2, buffer_size=10, replay_buffer_kwargs = rb_kwargs, 
#                 learning_rate = 1e-3, action_noise = noise, policy_kwargs = policy_kwargs)

# # Test the trained agent
# total_reward = 0
# num_episodes = 10  # Number of episodes to test
# for episode in range(num_episodes):
#     print("Episode:",)
#     observation, info = env.reset()
#     episode_reward = 0
#     terminated = False
#     while not terminated:
#         action, _states = model.predict(observation, deterministic=True)
#         observation, reward, terminated, truncated, info = env.step(action)
#         episode_reward += reward

#         if terminated or truncated:
#             total_reward += episode_reward
#             print(f"Episode {episode + 1}: Cumulative Reward = {episode_reward}")
#             break

# average_reward = total_reward / num_episodes
# print(f"Average Cumulative Reward over {num_episodes} episodes: {average_reward}")

