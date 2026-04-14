import pybullet as p
import pybullet_data
import numpy as np
import os
import time
import pickle
from robot import Panda
from tqdm import tqdm


# parameters
# Timestep for simulation updates (240Hz default)
control_dt = 1. / 240.

# create simulation and place camera
# Connect to the PyBullet physics engine using GUI connection for visual debugging
physicsClient = p.connect(p.GUI)
# Apply realistic gravity along the Z-axis
p.setGravity(0, 0, -9.81)
# Disable additional GUI elements for a cleaner view
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
# Set the initial viewpoint of the external observing camera 
p.resetDebugVisualizerCamera(cameraDistance=1.0, 
                                cameraYaw=40.0,
                                cameraPitch=-30.0, 
                                cameraTargetPosition=[0.5, 0.0, 0.2])

# load the objects
# Retrieve the native PyBullet data path containing base URDF models
urdfRootPath = pybullet_data.getDataPath()
# Insert standard ground plane ensuring things don't fall infinitely
plane = p.loadURDF(os.path.join(urdfRootPath, "plane.urdf"), basePosition=[0, 0, -0.625])
# Add a table acting as our interaction surface base
table = p.loadURDF(os.path.join(urdfRootPath, "table/table.urdf"), basePosition=[0.5, 0, -0.625])
# Add the target cube object to manipulate
cube = p.loadURDF(os.path.join(urdfRootPath, "cube_small.urdf"), basePosition=[0.5, 0, 0.025])
p.changeVisualShape(cube, -1, rgbaColor=[1, 0, 0, 1])      # change cube color to solid red

# load the robot
# Define the initial joint angles corresponding to the starting resting point or "home"
jointStartPositions = [0.0, 0.0, 0.0, -2*np.pi/4, 0.0, np.pi/2, np.pi/4, 0.0, 0.0, 0.04, 0.04]
# Initialize the custom Panda robot architecture with a lower resolution visual frame 
panda = Panda(basePosition=[0, 0, 0],
              baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
              jointStartPositions=jointStartPositions,
              cameraHeight=64,
              cameraWidth=64)

# collect the demonstrations
n_demos = 100
dataset = []
# Maximum step size per timestep movement increment
action_magnitude = 1.0
for demo_idx in tqdm(range(n_demos)):
    # reset the robot
    # Apply initial config state matching home
    panda.reset(jointStartPositions)
    # Give the testing cube a randomized starting position mapping within achievable bounding boxes 
    cube_position = np.random.uniform([0.3, -0.3, 0.025], [0.7, +0.3, 0.025])
    # Place the cube mapping without rotation constraints initially
    p.resetBasePositionAndOrientation(cube, cube_position, p.getQuaternionFromEuler([0, 0, 0]))
    # Establish intended goal target configuration (lifting cube vertically by 0.2 units)
    goal_position = cube_position + np.array([0., 0., 0.2])

    # internal frame counter for grasping
    counter = 0
    for time_idx in range(600):
        robot_state = panda.get_state()
        robot_pos = np.array(robot_state["ee-position"])
        cube_position, _ = p.getBasePositionAndOrientation(cube)

        # select the robot's action
        # Phase 1: Move towards the block until end-effector limits match bounds 
        if np.linalg.norm(robot_pos - cube_position) > 0.01 and counter < 100:
            action = cube_position - robot_pos
            if np.linalg.norm(action) > action_magnitude:
                action *= action_magnitude / np.linalg.norm(action)
            # Maintain open gripper state while driving to destination
            gripper_action = 1.
        # Phase 2: Close gripping mechanism explicitly holding position
        elif counter < 100:
            action = np.zeros(3)
            gripper_action = -1.
            counter += 1
        # Phase 3: Lift block upwards targeting goal
        else:
            action = goal_position - cube_position
            if np.linalg.norm(action) > action_magnitude:
                action *= action_magnitude / np.linalg.norm(action)
            # Enforce gripper locking actively 
            gripper_action = -1.

        # store the state-action pair
        state = robot_pos.tolist()
        dataset.append([robot_state["static"], robot_state["ee"], state + action.tolist() + [gripper_action]])

        panda.move_to_pose(robot_pos + action, ee_rotz=0, positionGain=0.01)
        if gripper_action < 0.:
            panda.close_gripper()
        else:
            panda.open_gripper()
            
        p.stepSimulation()
        time.sleep(control_dt)

# save the dataset of demonstrations
pickle.dump(dataset, open("dataset.pkl", "wb"))
print("dataset has this many state-action pairs:", len(dataset))
