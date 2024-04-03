from typing import Optional

import numpy as np

from panda_gym.envs.core import RobotTaskEnv
from panda_gym.envs.robots.panda import Panda
from panda_gym.pybullet import PyBullet

from test_reach_task_code import CustomReachTask

class CustomReachEnv(RobotTaskEnv):

    def __init__(
        self,
        render_mode: str = "rgb_array",
        reward_type: str = "dense",
        control_type: str = "ee",
        renderer: str = "Tiny",
        render_width: int = 720,
        render_height: int = 480,
        render_target_position: Optional[np.ndarray] = None,
        render_distance: float = 1.4,
        render_yaw: float = 45,
        render_pitch: float = -30,
        render_roll: float = 0,
    ) -> None:
        sim = PyBullet(render_mode=render_mode, renderer=renderer)
        robot = Panda(sim, block_gripper=True, base_position=np.array([-0.6, 0.0, 0.0]), control_type=control_type)
        task = CustomReachTask(sim, reward_type=reward_type, get_ee_position=robot.get_ee_position)
        super().__init__(
            robot,
            task,
            render_width=render_width,
            render_height=render_height,
            render_target_position=render_target_position,
            render_distance=render_distance,
            render_yaw=render_yaw,
            render_pitch=render_pitch,
            render_roll=render_roll,
        )
        
    def get_task_goal_position(self) -> np.ndarray:
        return self.task.get_goal_position()

env = CustomReachEnv(render_mode="human")

observation, info = env.reset()

for _ in range(10000):
    # action = env.action_space.sample() # random action
    action = env.get_task_goal_position() - env.robot.get_ee_position()
    # print(env.get_task_goal_position(), env.robot.get_ee_position())
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()