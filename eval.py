import os
import numpy as np
import torch
from tqdm import tqdm
import json
import copy
import random
import time
import pybullet as p
import pybullet_data

from policies import MLPPolicy, DiffusionPolicy, EMA
from utils import get_device, ObservationBuffer
from config import get_config
from robot import Panda


def evaluate(args):
    # The physical simulation timeline step duration
    control_dt = 1. / 240.

    # create simulation and place camera
    # Launch pybullet without a GUI visualizing the physics simulation graphically
    physicsClient = p.connect(p.GUI)
    # Applying realistic Earth surface gravity
    p.setGravity(0, 0, -9.81)
    # Remove GUI interface components 
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    # Frame visual angle coordinates representing standard static viewing setups 
    p.resetDebugVisualizerCamera(cameraDistance=1.0, 
                                    cameraYaw=40.0,
                                    cameraPitch=-30.0, 
                                    cameraTargetPosition=[0.5, 0.0, 0.2])

    # load the objects
    urdfRootPath = pybullet_data.getDataPath()
    # Add a flat physical bounding surface rendering plane collisions
    plane = p.loadURDF(os.path.join(urdfRootPath, "plane.urdf"), basePosition=[0, 0, -0.625])
    # Setup working table prop mapping
    table = p.loadURDF(os.path.join(urdfRootPath, "table/table.urdf"), basePosition=[0.5, 0, -0.625])
    # Insert target interaction cube block and save the tracker ID
    cube = p.loadURDF(os.path.join(urdfRootPath, "cube_small.urdf"), basePosition=[0.5, 0, 0.025])

    # load the robot
    # Base reset default joint posture array layout logic
    jointStartPositions = [0.0, 0.0, 0.0, -2*np.pi/4, 0.0, np.pi/2, np.pi/4, 0.0, 0.0, 0.04, 0.04]
    # Initialize the custom logic Panda class implementation instance managing URDF interactions 
    panda = Panda(basePosition=[0, 0, 0],
                    baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
                    jointStartPositions=jointStartPositions)

    # Automatically resolve optimal computation engine environment logic
    device = get_device(args.device)
    
    obs_shape = 3 + 128 + 128
    
    if args.policy == 'mlp':
        policy = MLPPolicy(state_dim=obs_shape,
                           action_dim=4,
                           hidden_dims=args.hidden_dims)
    elif args.policy == 'diffusion':
        policy = DiffusionPolicy(state_dim=obs_shape,
                                 action_dim=4,
                                 emb_dim=args.emb_dim,
                                 hidden_dims=args.hidden_dims,
                                 n_heads=args.n_heads,
                                 n_layers=args.n_layers,
                                 timesteps=args.timesteps,
                                 obs_horizon=args.obs_horizon,
                                 pred_horizon=args.pred_horizon,
                                 n_rollout_actions=args.n_rollout_actions,
                                 device=args.device)
    # Validate the load location has valid arguments
    assert args.loadloc is not None, f"No pretrained model found at location: {args.loadloc}"
    # Target best saved checkpoint
    load_model = os.path.join(args.loadloc, 'best_model.pt')

    print(f'Loading policy from {load_model}')
    # Safely load the state dict logic from weights mapping
    ckpt = torch.load(load_model, weights_only=True, map_location=device)
    
    # Switch models directly to Exponential Moving Averages 
    if args.use_ema_model:
        print('Using EMA policy for evaluation')
        ema_policy = copy.deepcopy(policy).to(device)
        state_dict = ckpt['ema_policy']
        ema_policy.load_state_dict(state_dict)
        eval_policy = ema_policy
    else:
        print('EMA not found, using standard policy')
        policy = policy.to(device)
        state_dict = ckpt['policy']
        policy.load_state_dict(state_dict)
        eval_policy = policy
        
    eval_policy.eval()

    # Pre-allocate tracking ring buffer containing state history 
    obs_buffer = ObservationBuffer(args)

    for eval in tqdm(range(args.num_evals), desc='Rolling out policy'):
        # reset the robot
        panda.reset(jointStartPositions)
        cube_position = np.random.uniform([0.3, -0.3, 0.025], [0.7, +0.3, 0.025])
        p.resetBasePositionAndOrientation(cube, cube_position, p.getQuaternionFromEuler([0, 0, 0]))
        p.changeVisualShape(cube, -1, rgbaColor=[1, 0, 0, 1])      # change cube color
        
        robot_state = panda.get_state()
        obs = {'observation': robot_state['ee-position'],
               'static_image': robot_state['static'],
               'ee_image': robot_state['ee']}
               
        # Empty observation history and populate with new state
        obs_buffer.reset()
        obs_buffer.add(obs)
        
        for t in range(0, args.time_horizon, args.n_rollout_actions):
            obs_sequence = obs_buffer.get_sequence()
            for k, v in obs_sequence.items():
                obs_sequence[k] = torch.Tensor(v)[None, :].to(device)
            action = eval_policy.get_action(obs_sequence)

            # iterate through action chunks
            for a_idx in range(args.n_rollout_actions):
                robot_state = panda.get_state()
                robot_pos = np.array(robot_state['ee-position'])
                panda.move_to_pose(robot_pos + action[a_idx, :3], ee_rotz=0, positionGain=0.01)
                
                # actuate gripper
                if action[a_idx, -1] < 0:
                    panda.close_gripper()
                else:
                    panda.open_gripper()

                p.stepSimulation()
                time.sleep(control_dt)

                robot_state = panda.get_state()
                obs = {'observation': robot_state['ee-position'],
                       'static_image': robot_state['static'],
                       'ee_image': robot_state['ee']}
                obs_buffer.add(obs)



if __name__ == '__main__':
    args = get_config()

    # impose random initialization constraints for repeatability  
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # variables that can be modified by user
    ignore_args = {'loadloc', 'num_evals', 'render_mode', 'device', 'save_video', 'domain_randomize', 'inference_steps', 'n_rollout_actions'}
    
    # load saved arguments
    trained_args_loc = os.path.join(args.loadloc, 'arguments.json')
    trained_args = json.load(open(trained_args_loc, 'r'))
    for k, v in trained_args.items():
        if k in ignore_args:
            continue
        if hasattr(args, k) and k != 'loadloc':
            setattr(args, k, v)
        else:
            print(f"Warning: Unrecognized arg '{k}' in {trained_args_loc}")
            
    args.time_horizon = 1000
    
    evaluate(args)