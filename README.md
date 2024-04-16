# Visual-Robot-Project

Setup:
Enter the following into the terminal upon installing the files:

Step 1 - git clone https://github.com/CASIA-IVA-Lab/FastSAM.git

Step 2 - "python -m venv FastSAM"

Step 3 - "FastSAM/scripts/activate" or "source venv/bin/activate" on mac

Step 4 - "pip install panda-gym stable_baselines3 stable-baselines3[extra] sb3_contrib opencv-python tensorboard tensorflow torchvision pycocotools matplotlib"

Step 5 - If using 'flip_sam_env' then enter "pip install git+https://github.com/facebookresearch/segment-anything.git" into console

Then it should work correctly

# Use file 'Interface.py' to run the main code with all the features.

To load the tensorboard logs through the console type: "python -m tensorboard.main --logdir=./logs/"

------------------------------------------------------------------
