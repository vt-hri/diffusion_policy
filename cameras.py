import pybullet as p
import numpy as np


# class for an external camera with a fixed base
# this camera does not move during the simulation
class ExternalCamera():

	def __init__(self, cameraDistance=1.6, cameraYaw=45, cameraPitch=-30, cameraRoll=0, cameraTargetPosition=[0,0,0], cameraWidth=256, cameraHeight=256):
		# Initialize camera parameters
		self.cameraDistance = cameraDistance
		self.cameraYaw = cameraYaw
		self.cameraPitch = cameraPitch
		self.cameraRoll = cameraRoll
		self.cameraTargetPosition = cameraTargetPosition
		self.cameraWidth = cameraWidth
		self.cameraHeight = cameraHeight

		# Compute the projection matrix for PyBullet
		self.proj_matrix = p.computeProjectionMatrixFOV(fov=60,
														aspect=self.cameraWidth / self.cameraHeight,
														nearVal=0.01,
														farVal=2.0)

		# Compute the view matrix based on camera pose
		self.view_matrix = p.computeViewMatrixFromYawPitchRoll(distance=self.cameraDistance,
															yaw=self.cameraYaw,
															pitch=self.cameraPitch,
															roll=self.cameraRoll,
															cameraTargetPosition=self.cameraTargetPosition,
															upAxisIndex=2)

	def get_image(self):
		# Grab the camera image from PyBullet utilizing the precomputed projection and view matrices
		_, _, rgba, _, _ = p.getCameraImage(width=self.cameraWidth,
												height=self.cameraHeight,
												viewMatrix=self.view_matrix,
												projectionMatrix=self.proj_matrix,
												renderer=p.ER_BULLET_HARDWARE_OPENGL,
												flags=p.ER_NO_SEGMENTATION_MASK)

		rgba = np.array(rgba, dtype=np.uint8).reshape((self.cameraWidth, self.cameraHeight, 4))
		rgb = rgba[:, :, :3]
		return rgb


# class for an onboard camera mounted to the robot's end-effector
# this camera moves with the robot during the simulation
class OnboardCamera():
	def __init__(self, cameraDistance=0.2, cameraOffsetPosition=[0.05, 0.0, 0.0], cameraOffsetQuaternion=p.getQuaternionFromEuler([0, -np.pi/2-np.pi/6, 0]), cameraWidth=256, cameraHeight=256):
		# Store the distance and default dimensions
		self.cameraDistance = cameraDistance
		self.cameraWidth = cameraWidth
		self.cameraHeight = cameraHeight
		# Save offsets used to calculate the relative physical position of the camera from the end-effector
		self.cameraOffsetPosition = cameraOffsetPosition
		self.cameraOffsetQuaternion = cameraOffsetQuaternion

		# Statically compute projection matrix based on fixed FOV specifications
		self.proj_matrix = p.computeProjectionMatrixFOV(fov=60,
														aspect=self.cameraWidth / self.cameraHeight,
														nearVal=0.01,
														farVal=2.0)

	def get_image(self, ee_position, ee_quaternion):
		# update the camera position and orientation as the robot moves
		# Multiply current robot end-effector coordinates by the camera's relative offsets
		cam_pos, cam_orn = p.multiplyTransforms(ee_position,
												ee_quaternion,
												self.cameraOffsetPosition,
												self.cameraOffsetQuaternion)
		# Calculate the rotation matrix to build the directional vectors
		rot_mat = np.array(p.getMatrixFromQuaternion(cam_orn)).reshape(3, 3)
		# Define mapping for where the camera is mathematically looking forward vs upwards
		cam_forward = rot_mat @ np.array([1, 0, 0])
		cam_up = rot_mat @ np.array([0, 0, 1])

		# Compute the dynamic view matrix frame using eye position, mathematical target, and up direction
		view_matrix = p.computeViewMatrix(cameraEyePosition=cam_pos,
											cameraTargetPosition=cam_pos + self.cameraDistance * cam_forward,
											cameraUpVector=cam_up)

		# Render and capture an image using PyBullet's hardware renderer based on up-to-date coordinate mappings
		_, _, rgba, _, _ = p.getCameraImage(width=self.cameraWidth,
												height=self.cameraHeight,
												viewMatrix=view_matrix,
												projectionMatrix=self.proj_matrix,
												renderer=p.ER_BULLET_HARDWARE_OPENGL,
												flags=p.ER_NO_SEGMENTATION_MASK)

		rgba = np.array(rgba, dtype=np.uint8).reshape((self.cameraWidth, self.cameraHeight, 4))
		rgb = rgba[:, :, :3]
		return rgb
