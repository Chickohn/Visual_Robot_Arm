# Manual Control Imports
import re
import tkinter as tk
from PIL import Image, ImageTk
from threading import Thread
import gymnasium as gym
import panda_gym
import queue

# ML Imports
import os
import numpy as np
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
# from stable_baselines3.common.logger import configure
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3 import HerReplayBuffer
from stable_baselines3 import DDPG
from tensorboard import program
import socket
import webbrowser
from datetime import datetime

def parse_action(input_command, env):
        """
        Use regex to break down the command into three parts: 
        - command: Direction of movement or specific command
        - value: Value to set the movement action to. (divided by 10)
        - close_gripper: Whether or not to close the gripper or not (defined by the presence of a 'c' at the end of the command)
        """
        action = [0, 0, 0, 0]

        match = re.match(r"([a-z]+)(\d+)?(c)?", input_command)
        if match:
            command, value, close_gripper = match.groups()
            value = float(value) if value else 5

            value /= 10

            if command == "d":
                action[2] = -value
            elif command == "u":
                action[2] = value
            elif command == "o":
                action[3] = value
            elif command == "c" and not close_gripper:
                action[3] = -value
            elif command == "b":
                action[0] = -value
            elif command == "f":
                action[0] = value
            elif command == "l":
                action[1] = value
            elif command == "r":
                action[1] = -value
            elif command == "lift":
                action[2] = value
                action[3] = -value
            elif command == "s":
                env.render_camera()
            
            if close_gripper:
                action[3] = -0.5

        return action

def setup_initial_screen(restart=False):
    """
    Sets up the initial GUI. 
    - restart: bool = False (Called True when returning to the main menu from a different screen)
    """
    # Clear the window
    for widget in root.winfo_children():
        widget.destroy()

    # Create buttons for the initial screen
    button_control = tk.Button(root, text="Control Manually", command=setup_manual_control_screen)
    button_control.pack(pady=20)

    button_without_vision = tk.Button(root, text="Without Vision", command=lambda: setup_training_screen(vision=False))
    button_without_vision.pack(pady=20)

    button_with_vision = tk.Button(root, text="With Vision", command=setup_training_screen)
    button_with_vision.pack(pady=20)

    button_with_vision = tk.Button(root, text="Load Model", command=load_model)
    button_with_vision.pack(pady=20)

    root.update_idletasks()
    root.geometry(f"{root.winfo_reqwidth()+250}x{root.winfo_reqheight()+20}")

    if restart:
        command_queue.put("stop") # Closes any open environments

def setup_manual_control_screen():
    """
    Sets up the GUI for manual control.
    """
    def update_images():
        """
        Updates the images on the GUI.
        """
        new_input_image = Image.open("Output/input.jpg")  # Adjust path as needed
        new_output_image = Image.open("Output/output.jpg")  # Adjust path as needed

        # Update PhotoImage objects
        input_photo = ImageTk.PhotoImage(new_input_image.resize((250, 250)))
        output_photo = ImageTk.PhotoImage(new_output_image.resize((250, 250)))

        # Update the label images
        input_image_label.configure(image=input_photo)
        output_image_label.configure(image=output_photo)

        # Keep references to the images to prevent garbage collection
        input_image_label.image = input_photo
        output_image_label.image = output_photo


    # Clear the window
    for widget in root.winfo_children():
        widget.destroy()

    # Create entry field for command input
    entry = tk.Entry(root)
    entry.pack(pady=10)

    # Function to handle command input
    def handle_command():
        command = entry.get()
        print(f"Command received: {command}")
        command_queue.put(command)
        entry.delete(0, tk.END)
        update_images()

    # Button to submit command
    submit_button = tk.Button(root, text="Submit Command", command=handle_command)
    submit_button.pack(pady=10)

    # Button to return to main menu
    back_button = tk.Button(root, text="Back to Main Menu", command=lambda: setup_initial_screen(restart=True))
    back_button.pack(pady=10)

    # Create a frame for the input section
    input_frame = tk.Frame(root)
    input_frame.pack(side=tk.LEFT, padx=20, pady=20)

    # Label for the input
    input_label = tk.Label(input_frame, text="Input")
    input_label.pack()

    # Load and display the input image
    input_image = Image.open("Output/input.jpg")  # Replace with your image file
    input_photo = ImageTk.PhotoImage(input_image.resize((250, 250)))
    input_image_label = tk.Label(input_frame, image=input_photo)
    input_image_label.pack()

    # Create a frame for the output section
    output_frame = tk.Frame(root)
    output_frame.pack(side=tk.RIGHT, padx=20, pady=20)

    # Label for the output
    output_label = tk.Label(output_frame, text="Output")
    output_label.pack()

    # Assuming you have an output image, load and display it
    output_image = Image.open("Output/output.jpg")  # Replace with your image file
    output_photo = ImageTk.PhotoImage(output_image.resize((250, 250)))
    output_image_label = tk.Label(output_frame, image=output_photo)
    output_image_label.pack()

    # Keep references to the images to prevent garbage collection
    input_image_label.image = input_photo
    output_image_label.image = output_photo

    root.update_idletasks()
    root.geometry(f"{root.winfo_reqwidth()}x{root.winfo_reqheight()}")

    # Start the gym environment in a new thread to avoid blocking the GUI
    Thread(target=manual_control, daemon=True).start()

def manual_control():
    try:
        env = gym.make('PandaPickAndPlace-v3', render_mode="human")
        env.reset()
        env.render_camera()

        while True:
            try:
                # Get a command from the queue
                command = command_queue.get(timeout=1)  # Adjust timeout as needed
                print(f"Processing command: {command}")
                action = parse_action(command, env)
                if command == "camera":
                    camera1 = float(input("pos1: "))
                    camera2 = float(input("pos2: "))
                    camera3 = float(input("pos3: "))
                    camera_pos = [camera1, camera2, camera3]
                    env.render_camera(camera_pos)
                
                if command == "stop":
                    env.close()
                    break
                observation, reward, terminated, truncated, info = env.step(action)
                
                if terminated or truncated:
                    observation, info = env.reset()
            except queue.Empty:
                continue
    except Exception as e:
        print(f"Error in manual_control: {e}")  

def setup_training_screen(vision: bool = True):
    def update_screen(timesteps: int, vision: bool):
        for widget in root.winfo_children():
            widget.destroy()

        label = tk.Label(root, text="Training...")
        label.pack(pady=10)

        Thread(target=lambda: train(timesteps=timesteps, vision=vision), daemon=True).start()

        back_button = tk.Button(root, text="Back to Main Menu", command=lambda: setup_initial_screen(restart=True))
        back_button.pack(pady=10)

    def train(timesteps: int, vision: bool):

        env = gym.make('PandaPickAndPlace-v3', render_mode="human", vision=vision)

        env = make_vec_env(lambda: env, n_envs=1)

        n_actions = env.action_space.shape[-1]
        action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=1.1 * np.ones(n_actions))


        # Create the DDPG model
        model = DDPG("MultiInputPolicy", 
                    env, 
                    action_noise=action_noise, 
                    replay_buffer_class=HerReplayBuffer,
                    verbose=1, 
                    batch_size= 512,
                    buffer_size=50_000, 
                    learning_rate=0.001, 
                    tensorboard_log=log_dir
                    )
        
        load_tensorboard()

        model.learn(total_timesteps=timesteps)

        mean_reward, std_reward = evaluate_policy(model, model.get_env(), return_episode_rewards=False, n_eval_episodes=25)

        for widget in root.winfo_children():
            widget.destroy()

        label = tk.Label(root, text="Training Results:")
        label.pack(pady=10, padx=10)

        mean_label = tk.Label(root, text=f"Mean reward: {mean_reward}")
        mean_label.pack()

        std_label = tk.Label(root, text=f"Std reward: {std_reward}")
        std_label.pack()

        save_button = tk.Button(root, text="Save Model", command=lambda: save_model(model, env))
        save_button.pack(pady=10)

        back_button = tk.Button(root, text="Back to Main Menu", command=lambda: setup_initial_screen(restart=True))
        back_button.pack(pady=10)

        root.update_idletasks()
        root.geometry(f"{root.winfo_reqwidth()}x{root.winfo_reqheight()}")




    # Clear the window
    for widget in root.winfo_children():
        widget.destroy()

    label = tk.Label(root, text= "Input the number of timesteps to train for:")
    label.pack(pady=10)
    # Create entry field for command input
    entry = tk.Entry(root)
    entry.pack(pady=10)

    submit_button = tk.Button(root, text="Submit", command=lambda: update_screen(timesteps=int(entry.get()), vision=vision))
    submit_button.pack(pady=10)

def load_model():

    # Clear the window
    for widget in root.winfo_children():
        widget.destroy()

    # Get list of model files in the 'models' directory
    models_dir = 'models'
    try:
        model_files = os.listdir(models_dir)
    except FileNotFoundError:
        model_files = []
        print("The 'models' directory does not exist.") 
        
    # Create a Tkinter string variable to hold the selected model
    model_var = tk.StringVar(root)
    model_var.set(model_files[0] if model_files else "No models available")

    # Create a dropdown menu
    model_menu = tk.OptionMenu(root, model_var, *model_files)
    model_menu.pack(pady=20)

    # Create a button to confirm the model selection
    choose_button = tk.Button(root, text="Choose Model", command=lambda: chosen_model(model=model_var.get()))
    choose_button.pack(pady=10)

    back_button = tk.Button(root, text="Back", command=lambda: setup_initial_screen(restart=True))
    back_button.pack(pady=10)
    
def save_model(model, env):
    # for widget in root.winfo_children():
    #     widget.destroy()
    save_screen = tk.Tk()
    save_screen.title("Save your model")
    save_screen.geometry("200x200")

    name_label = tk.Label(save_screen, text="Name the model:")
    name_label.pack(pady=10)

    name_entry = tk.Entry(save_screen)
    name_entry.pack()

    name_button = tk.Button(save_screen, text="Save", command=lambda: save(model, name_entry.get(), env))
    name_button.pack()

    def save(model, model_name, env):
        # if model_name == "":
        #     model.save("models/")
        # model.save("models/"+ model_name)
        directory = "models/"
        if not os.path.exists(directory):
            os.makedirs(directory)

        if model_name == "":
            model_name = "DDPG_model"
        
        # Initialize the base filename
        filename = os.path.join(directory, model_name)
        
        # Check if the file exists and modify it if necessary
        counter = 1
        original_filename = filename
        while os.path.exists(filename + ".zip"):
            filename = f"{original_filename}_{counter}"
            counter += 1

        # Save the model
        model.save(filename)

        save_screen.destroy()

        env.close()
        
def chosen_model(model):

    # Clear the window
    for widget in root.winfo_children():
        widget.destroy()

    episodes_label = tk.Label(root, text="Input number of episodes to test:")
    episodes_label.pack(pady=10)

    entry = tk.Entry(root)
    entry.pack(pady=10)

    submit_button = tk.Button(root, text="Submit", command=lambda: test(int(entry.get()), model))
    submit_button.pack(pady=10)

    print(command_queue)

    def test(episodes, model):
        error_label = tk.Label(root, text="Loading Environment...")
        error_label.pack()

        env = gym.make('PandaPickAndPlace-v3', render_mode="human", vision=False)
        env = make_vec_env(lambda: env, n_envs=1)
        model = DDPG.load("models/"+model, env=env)

        for widget in root.winfo_children():
            widget.destroy()

        error_label = tk.Label(root, text="Testing Model...")
        error_label.pack()

        total_reward = 0
        num_episodes = episodes  # Number of episodes to test
        episode_rewards = {}
        for episode in range(num_episodes):
            observation = env.reset()
            episode_reward = 0
            terminated = False
            while not terminated:
                action, _states = model.predict(observation, deterministic=True)
                observation, reward, done, info = env.step(action)
                episode_reward += reward

                if done:
                    total_reward += episode_reward
                    episode_rewards[f"Episode {episode + 1}"] = episode_reward
                    break
        
        for widget in root.winfo_children():
            widget.destroy()
        
        episode_var = tk.StringVar(root)
        episode_var.set("Select an Episode")

        episode_menu = tk.OptionMenu(root, episode_var, *episode_rewards.keys())
        episode_menu.pack(pady=10)

        reward_label = tk.Label(root, text="")
        reward_label.pack()

        def update_reward_label(*args):
            episode = episode_var.get()
            reward = episode_rewards.get(episode, "No data")
            reward_label.config(text=f"Cumulative Reward: {reward}")
            reward_label.config(pady=10)

        # Update the label whenever the selection changes
        episode_var.trace_add("write", update_reward_label)

        average_reward = total_reward / num_episodes
        evaluation_label = tk.Label(root, text=f"Average Cumulative Reward over {num_episodes} episodes:\n{average_reward}")
        evaluation_label.pack(pady=10)

        submit_button = tk.Button(root, text="Open TensorBoard", command= load_tensorboard)
        submit_button.pack(pady=10)

        back_button = tk.Button(root, text="Back to Main Menu", command=lambda: setup_initial_screen(restart=True))
        back_button.pack(pady=10)

        env.close()

def check_port(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

def load_tensorboard(port=6006):
    if not check_port(port):
        tb = program.TensorBoard()
        tb.configure(argv=[None, '--logdir', log_dir, '--port', str(port)])
        url = tb.launch()
    webbrowser.open(url, new=1)

# Main Loop
command_queue = queue.Queue()
tensorboard_running = False
log_dir = "./logs/"
os.makedirs(log_dir, exist_ok=True)

root = tk.Tk()
root.title("Visual Robot Project GUI")
root.geometry("300x300")

setup_initial_screen()
root.mainloop()