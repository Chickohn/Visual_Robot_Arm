from typing import Any, Dict, Optional
import numpy as np
from panda_gym.envs.core import Task, RobotTaskEnv
from panda_gym.envs.robots.panda import Panda
from panda_gym.pybullet import PyBullet
from panda_gym.utils import distance
from pathlib import Path
import os
import time

class CustomReachTask(Task):
    def __init__(
        self,
        sim,
        get_ee_position,
        reward_type="sparse",
        distance_threshold=0.05,
        goal_range=0.3,
        beaker_files_folder="Beaker_Files",
    ) -> None:
        super().__init__(sim)
        self.reward_type = reward_type
        self.distance_threshold = distance_threshold
        self.get_ee_position = get_ee_position
        self.goal_range_low = np.array([-goal_range / 2, -goal_range / 2, 0.005])
        self.goal_range_high = np.array([goal_range / 2, goal_range / 2, 0.005])
        self.beaker_files_folder = beaker_files_folder
        self.beaker_files = ["Beaker_500ml.obj"] # "beaker_250ml_inst.usda", "beaker_250ml.usda", "beaker_500ml_inst_mesh.usda", "beaker_500ml_inst.usda", "beaker_500ml.usda", "beaker_500ml.usd"]
        self.current_beaker_id = None
        self.prev_distance_to_goal = None
        with self.sim.no_rendering():
            self._create_scene()

    def _create_scene(self) -> None:
        self.sim.create_plane(z_offset=-0.4)
        self.sim.create_table(length=1.1, width=0.7, height=0.4, x_offset=-0.3)
        self.sim.create_sphere(
            body_name="target",
            radius=0.02,
            mass=0.0,
            ghost=True,
            position=np.zeros(3),
            rgba_color=np.array([0.1, 0.9, 0.1, 0.3]),
        )
        self.current_beaker_id = self.sim.loadURDF(fileName="Beaker_500ml.urdf", body_name="object")

    def get_obs(self) -> np.ndarray:
        object_position = self.sim.get_base_position("object")
        object_rotation = self.sim.get_base_rotation("object")
        object_velocity = self.sim.get_base_velocity("object")
        self.sim.set_base_pose("target", object_position + np.array([-0.06, 0.0, 0.13]), np.array([1.0, 0.0, 0.0, 1.0]))
        # print("object position: ", object_position, "\n", object_rotation, "\n", object_velocity, "\n target position: ", self.sim.get_base_position("target")-object_position)
        object_angular_velocity = self.sim.get_base_angular_velocity("object")
        observation = np.concatenate([object_position, object_rotation, object_velocity, object_angular_velocity])
        return observation

    def get_achieved_goal(self) -> np.ndarray:
        target_position = np.array(self.sim.get_base_position("target"))
        return target_position

    def reset(self) -> None:
        super().reset()
        self.goal = self._sample_goal()
        object_position = self._sample_object()
        self.sim.set_base_pose("object", object_position, np.array([1.0, 0.0, 0.0, 1.0]))
        self.prev_distance_to_goal = np.linalg.norm(self.goal - self.get_achieved_goal())

    def _sample_goal(self) -> np.ndarray:
        """Randomize goal."""
        goal = self.np_random.uniform(self.goal_range_low, self.goal_range_high)
        return goal
    
    def _sample_object(self) -> np.ndarray:
        """Randomize start position of object."""
        object_position = np.array([0.0, 0.0, 0.0])
        noise = self.np_random.uniform(self.goal_range_low, self.goal_range_high)
        object_position += noise
        return object_position
    
    def get_goal_position(self) -> np.ndarray:
        return self.goal
    
    def is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> np.ndarray:
        d = distance(achieved_goal, desired_goal)
        return np.array(d < self.distance_threshold, dtype=bool)

    def compute_reward(self, achieved_goal, desired_goal, info: Dict[str, Any]) -> np.ndarray:
        if self.is_success(achieved_goal, desired_goal):
            return 1.0
        else:
            current_distance = np.linalg.norm(desired_goal - achieved_goal)
            
            if self.reward_type == "dense":
                reward = self.prev_distance_to_goal - current_distance
            else:
                reward = -np.array(current_distance > self.distance_threshold, dtype=np.float32)
            
            self.prev_distance_to_goal = current_distance
            
            return np.array(reward, dtype=np.float32)
            # d = distance(achieved_goal, desired_goal)
            # if self.reward_type == "sparse":
            #     return -np.array(d > self.distance_threshold, dtype=np.float32)
            # else:
            #     return -d.astype(np.float32)

class CustomReachEnv(RobotTaskEnv):

    def __init__(
        self,
        render_mode: str = "rgb_array",
        reward_type: str = "sparse",
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

# env = CustomReachEnv(render_mode="human")

# observation, info = env.reset()

# for _ in range(1000):
#     action = env.action_space.sample() # random action
#     # action = env.get_task_goal_position() - env.robot.get_ee_position()
#     # print(env.get_task_goal_position(), env.robot.get_ee_position())
#     observation, reward, terminated, truncated, info = env.step(action)

#     if terminated or truncated:
#         observation, info = env.reset()