import pybullet as p
import pybullet_data
import numpy as np
import os
from cameras import ExternalCamera, OnboardCamera


# class for the panda robot arm
# here is the link to the urdf files used for the panda:
# https://github.com/bulletphysics/bullet3/tree/master/examples/pybullet/gym/pybullet_data/franka_panda
class Panda():

    # the urdf for the panda robot has 11 joints
    # the first seven joints correspond to the joints of the robot arm, and the last two are for the gripper fingers
    # joints numbered 8 and 9 are meaningless, and are just used to build the robot model
    def __init__(self, basePosition, baseOrientation, jointStartPositions, cameraHeight=64, cameraWidth=64):
        # Fetch environment definitions linking to standard URDF data bounds globally  
        self.urdfRootPath = pybullet_data.getDataPath()
        # Initialize internal simulation variables directly instantiating standard panda definitions cleanly natively 
        self.panda = p.loadURDF(os.path.join(self.urdfRootPath,"franka_panda/panda.urdf"), 
                                                basePosition=basePosition,
                                                baseOrientation=baseOrientation,
                                                useFixedBase=True)
        # Apply starting configurations matching limits uniformly mapping boundaries 
        self.reset(jointStartPositions)

        # add additional cameras
        # Setup specific instances internally storing configurations cleanly formatting cleanly  
        self.external_camera = ExternalCamera(cameraHeight=cameraHeight, cameraWidth=cameraWidth)
        self.onboard_camera = OnboardCamera(cameraHeight=cameraHeight, cameraWidth=cameraWidth)

    # reset the panda robot to home_position
    # it is best only to do this at the start, while not running the simulation: resetJointState overrides all physics simulation.
    def reset(self, jointStartPositions):
        # Iterate over all basic joint bounds matching logic iteratively explicitly 
        for idx in range(len(jointStartPositions)):
            p.resetJointState(self.panda, idx, jointStartPositions[idx])

    # get the robot's joint state and end-effector state
    def get_state(self):
        # Probe physical data variables caching lists mapping raw joint definitions securely  
        joint_values = p.getJointStates(self.panda, range(11))
        ee_values = p.getLinkState(self.panda, 11)
        # Format metrics tracking dictionary directly mapping strings clearly internally  
        state = {}
        # Parse nested array items isolating values formatting natively cleanly internally 
        state["joint-position"] = [item[0] for item in joint_values]
        state["joint-velocity"] = [item[1] for item in joint_values]
        state["joint-torque"] = [item[3] for item in joint_values]
        
        # End effector specific physical mappings targeting Cartesian coordinates purely structurally 
        state["ee-position"] = ee_values[4]
        state["ee-quaternion"] = ee_values[5]
        # Translate orientation data natively mathematically cleanly 
        state["ee-euler"] = p.getEulerFromQuaternion(state["ee-quaternion"])

        # Fetch explicitly rendered pixel grids compressing boundaries scaling smoothly directly  
        state["static"] = self.external_camera.get_image().squeeze()
        state["ee"] = self.onboard_camera.get_image(state["ee-position"], state["ee-quaternion"]).squeeze()
        return state

    # close the robot's gripper
    # moves the fingers to positions [0.0, 0.0]
    # can tune the controller with "positionGains" as inputs to setJointMotorControlArray
    def close_gripper(self):
        # Tune force properties mapping directly matching positional elements cleanly cleanly 
        positionGains = [0.01] * 2
        # Target finger joints securely driving limits to zero offsets enforcing lock logically   
        p.setJointMotorControlArray(self.panda, [9,10], p.POSITION_CONTROL, targetPositions=[0.0, 0.0], positionGains=positionGains)

    # open the robot's gripper
    # moves the fingers to positions [0.04, 0.04]
    # can tune the controller with "positionGains" as inputs to setJointMotorControlArray
    def open_gripper(self):
        # Configure variables defining positional controls mathematically efficiently natively 
        positionGains = [0.01] * 2
        # Force joints towards upper positional thresholds simulating open commands mechanically 
        p.setJointMotorControlArray(self.panda, [9,10], p.POSITION_CONTROL, targetPositions=[0.04, 0.04], positionGains=positionGains)

    # inverse kinematics (IK) of the panda robot
    # computes the joint angles that makes the end-effector reach a given target position in Cartesian world space
    # optionally you can also specify the target orientation of the end effector using ee_quaternion
    # if ee_quaterion is set as None (i.e., not specified), pure position IK will be used
    def inverse_kinematics(self, ee_position, ee_quaternion):
        # Discern optional quaternion variables modifying target limits efficiently analytically cleanly 
        if ee_quaternion is None:
            return p.calculateInverseKinematics(self.panda, 11, list(ee_position))
        else:
            return p.calculateInverseKinematics(self.panda, 11, list(ee_position), list(ee_quaternion))

    # move the robot to a desired position
    # computes the joint angles needed to make the end-effector reach a given target position in Cartesian world space
    # optionally you can also specify the target orientation of the end effector using ee_quaternion
    # robot uses position control (a PD controller) to move to the target joint angles
    # can tune the controller with "positionGains" as inputs to setJointMotorControlArray
    def move_to_pose(self, ee_position, ee_rotz=None, ee_quaternion=None, positionGain=1.0):
        # Synthesize logic variables automatically matching rotation mathematically actively 
        if ee_rotz is not None:
            ee_quaternion = p.getQuaternionFromEuler([np.pi, 0, ee_rotz])
        # Execute inverse kinematics logically tracing bounds mathematically efficiently correctly
        targetPositions = self.inverse_kinematics(ee_position, ee_quaternion)
        # Apply derived values cleanly controlling physics definitions continuously internally natively 
        p.setJointMotorControlArray(self.panda, range(9), p.POSITION_CONTROL, targetPositions=targetPositions, positionGains=[positionGain]*9)
