# Visual-Robot-Project

Setup:
Enter the following into the terminal upon installing the files:
Step 1 - "python -m venv venv"

Step 2 - "venv/scripts/activate" or "source venv/bin/activate" on mac

Step 3 - "pip install panda-gym stable_baselines3 stable-baselines3[extra] sb3_contrib opencv-python tensorboard tensorflow torchvision opencv-python pycocotools matplotlib onnxruntime onnx"

Step 4 - "pip install git+https://github.com/facebookresearch/segment-anything.git"

Then it should work correctly

To load the tensorboard logs: "tensorboard --logdir=./logs/"

------------------------------------------------------------------

Notes for meeting:

- Having issues getting the test reach task to reset.
- Implemented SAM into test_flip_environment_code
- 'SAM Output Example.png' is the result of 2 minutes of loading - not sure if it is viable unless i just do it at the start, although it would take days to train even then
- Set the target to the lip of the beaker and should have set the goal as that target, but not sure how to check if it is reaching the goal and how it restarts


mobile SAM/ Fast SAM
erikaa wang panda robot task github


Completed:
- Set up a working camera that sits at the first joint of the robot arm for now
- Recreated a basic flip task
- Tested some Image Segmentation
- Looked at dense rewards in PandaFlipDense-v3 briefly
- tensorboard

To do:
(In the next two weeks):
- Integrate TensorFlow into the training
- Figure out how to specify dense rewards to teach robot to move to the right place and pick it up, and then rotate correctly
- Decide on how the camera will function, i.e. What position and where it will look, as well as what method of visual perception I will need to use to identify beakers

(In general):
- SegmentAnything (Meta)
- Create beaker URDFs and any other objects I might need (In Blender)
- Create a working method for training the robot via dense rewards (or something else if I find a better method)
- Train the robot to look at the camera data instead and be able to pick things up using that data instead
- Store a trained model
- Test the model on the new environment containing the beakers
