import gymnasium as gym
import panda_gym
import pybullet as p
from time import sleep
from panda_gym.envs.core import RobotTaskEnv
from panda_gym.envs.robots.panda import Panda
from panda_gym.envs.tasks.pick_and_place import PickAndPlace
from panda_gym.envs.core import Task
from panda_gym.pybullet import PyBullet
from typing import Optional
import cv2
import numpy as np
import matplotlib.pyplot as plt
from fastsam import FastSAM, FastSAMPrompt
import torch


class CameraBeakerEnv(RobotTaskEnv):
    """My robot-task environment."""

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
        robot = Panda(sim, block_gripper=False, base_position=np.array([-0.6, 0.0, 0.0]), control_type= control_type)
        task = PickAndPlace(sim, 
                            reward_type=reward_type, 
                            get_ee_position=robot.get_ee_position, 
                            gripper_width=robot.get_fingers_width,
                            inverse_kinematics=robot.inverse_kinematics
                            )
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

    def render_camera(self):
        print("1st")
        joint_pos, joint_ori = p.getLinkState(0, linkIndex=1)[:2]
        # Define the camera settings
        offset = 0.2  # Raise the camera 20cm above the joint
        camera_pos = [joint_pos[0], joint_pos[1], joint_pos[2] + offset]
        # print(camera_pos)
        # camera_pos = [1, 1, 1]  # Example position
        target_pos = [0, 0, 0]  # Point the camera looks at
        up_vector = [0, 0, 1]
        print("second")
        view_matrix = p.computeViewMatrix(camera_pos, target_pos, up_vector)

        # Define Field of View and other camera parameters
        fov = 60
        aspect = 1
        near_val = 0.5
        far_val = 3
        print("third")
        projection_matrix = p.computeProjectionMatrixFOV(fov, aspect, near_val, far_val)

        # Get the camera image
        width, height, rgb_img, depth_img, seg_img = p.getCameraImage(width=640, height=480, viewMatrix=view_matrix, projectionMatrix=projection_matrix)
        print("fourth")
        # Convert the image to a format OpenCV can use and show it
        np_img = np.reshape(rgb_img, (height, width, 4))
        np_img = np_img[:, :, :3].astype(np.uint8)
        frame = cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB)
        cv2.imwrite('C:/Users/fredd/Desktop/Dissertation/Visual_Robot_Arm/Output/input.jpg', frame)
        plt.imshow(frame)
        print("pre mask")

        model = FastSAM('./weights/FastSAM-s.pt')
        IMAGE_PATH = 'C:/Users/fredd/Desktop/Dissertation/Visual_Robot_Arm/Output/input.jpg'
        DEVICE = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        print("pre everything results")
        print(IMAGE_PATH, DEVICE)
        everything_results = model(IMAGE_PATH, device=DEVICE, retina_masks=True, imgsz=1024, conf=0.4, iou=0.9,)
        prompt_process = FastSAMPrompt(IMAGE_PATH, everything_results, device=DEVICE)

        # everything prompt
        # ann = prompt_process.everything_prompt()
        # ann = prompt_process.box_prompt(bboxes=[[200, 200, 300, 300]])
        ann = prompt_process.text_prompt(text="a beaker on a table")
        print(ann)

        prompt_process.plot(annotations = ann, output_path ='C:/Users/fredd/Desktop/Dissertation/Visual_Robot_Arm/Output/output.jpg')
        print("post mask")
        cv2.waitKey(1)

env = CameraBeakerEnv(render_mode="human")

observation, info = env.reset()
for _ in range(1000):
    action = env.action_space.sample()
    try:
        observation, reward, terminated, truncated, info = env.step(action)
    except:
        break

    # Render camera image
    env.render_camera()

    if terminated or truncated:
        observation, info = env.reset()

cv2.destroyAllWindows()