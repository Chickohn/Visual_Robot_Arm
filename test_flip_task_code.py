import gymnasium as gym
import panda_gym
import numpy as np
from stable_baselines3 import DDPG
from time import sleep
from typing import Any, Dict, Tuple
from typing import Optional
import numpy as np
from scipy.spatial.transform import Rotation as R
from panda_gym.envs.core import RobotTaskEnv
from panda_gym.envs.robots.panda import Panda
from panda_gym.envs.core import Task
from panda_gym.pybullet import PyBullet
from panda_gym.utils import angle_distance
from panda_gym.utils import distance
import pybullet as p

class MyTask(Task):
    def __init__(self, sim, reward_type: str = "dense", obj_xy_range: float = 0.3,):
        super().__init__(sim)
        self.reward_type = reward_type
        self.object_size = 0.04
        self.obj_range_low = np.array([-obj_xy_range / 2, -obj_xy_range / 2, 0])
        self.obj_range_high = np.array([obj_xy_range / 2, obj_xy_range / 2, 0])
        with self.sim.no_rendering():
            self._create_scene()

    def _create_scene(self) -> None:
        """Create the scene."""
        self.sim.create_plane(z_offset=-0.4)
        self.sim.create_table(length=1.1, width=0.7, height=0.4, x_offset=-0.3)
        self.sim.create_box(
            body_name="object",
            half_extents=np.ones(3) * self.object_size / 2,
            mass=1.0,
            ghost=False,
            position=np.array([0.0, 0.0, self.object_size / 2]),
            rgba_color=np.array([1.0, 1.0, 1.0, 0.5]),
            texture="colored_cube.png",
        )
        self.sim.create_box(
            body_name="target",
            half_extents=np.ones(3) * self.object_size / 2,
            mass=0.0,
            ghost=True,
            position=np.array([0.0, 0.0, 3 * self.object_size / 2]),
            rgba_color=np.array([1.0, 1.0, 1.0, 0.5]),
            texture="colored_cube.png",
        )

    def reset(self):
        # self.goal = R.random().as_quat()
        # object_position, object_orientation = self._sample_object()
        # self.sim.set_base_pose("target", np.array([0.0, 0.0, 3 * self.object_size / 2]), self.goal)
        # self.sim.set_base_pose("object", object_position, object_orientation)
        # randomly sample a goal position
        self.goal = np.random.uniform(-10, 10, 3)
        # reset the position of the object
        self.sim.set_base_pose("object", position=np.array([0.0, 0.0, 0.0]), orientation=np.array([1.0, 0.0, 0.0, 0.0]))

    def _sample_object(self) -> Tuple[np.ndarray, np.ndarray]:
        """Randomize start position of object."""
        object_position = np.array([0.0, 0.0, self.object_size / 2])
        noise = self.np_random.uniform(self.obj_range_low, self.obj_range_high)
        object_position += noise
        object_rotation = np.zeros(3)
        return object_position, object_rotation

    def get_obs(self):
        # the observation is the position of the object
        object_position = self.sim.get_base_position("object")
        object_rotation = self.sim.get_base_rotation("object", "quaternion")
        object_velocity = self.sim.get_base_velocity("object")
        object_angular_velocity = self.sim.get_base_angular_velocity("object")
        observation = np.concatenate([object_position, object_rotation, object_velocity, object_angular_velocity])
        return observation

    def get_achieved_goal(self):
        # the achieved goal is the current position of the object
        achieved_goal = self.sim.get_base_position("object")
        return achieved_goal

    def is_success(self, achieved_goal, desired_goal, info={}):  # info is here for consistancy
        # compute the distance between the goal position and the current object position
        d = distance(achieved_goal, desired_goal)
        # return True if the distance is < 1.0, and False otherwise
        return np.array(d < 1.0, dtype=bool)

    def compute_reward(self, achieved_goal, desired_goal, info={}):  # info is here for consistancy
        # for this example, reward = 1.0 if the task is successfull, 0.0 otherwise
        return self.is_success(achieved_goal, desired_goal, info).astype(np.float32)