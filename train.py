import os
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import random
import copy
import json

from policies import MLPPolicy, DiffusionPolicy, EMA
from utils import get_device
from data import Data, DataSequence
from config import get_config


def train(args):
    device = get_device(args.device)
    task_name = 'PickAndPlace'
    
    saveloc = os.path.join(os.getcwd(), args.saveloc, task_name, args.savename)
    os.makedirs(saveloc, exist_ok=True)

    # Dump initialization variables
    arguments = vars(args)
    with open(os.path.join(saveloc, 'arguments.json'), "w") as f:
        json.dump(arguments, f, indent=4)

    obs_shape = 3 + 128 + 128

    if args.policy == 'mlp':
        dataset = Data(args.dataset_path)
        policy = MLPPolicy(state_dim=obs_shape,
                           action_dim=4,
                           hidden_dims=args.hidden_dims)
    elif args.policy == 'diffusion':
        dataset = DataSequence(args.dataset_path,
                               obs_horizon=args.obs_horizon,
                               pred_horizon=args.pred_horizon)
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
    policy = policy.to(device)
    optimizer = Adam(policy.parameters(), lr=args.learning_rate)

    dataloader = DataLoader(dataset,
                            shuffle=True,
                            batch_size=args.batch_size)

    if args.use_ema_model:
        ema_policy = copy.deepcopy(policy).to(device)
        ema = EMA(ema_policy,
                  update_after_step=args.ema_update_after_step,
                  inv_gamma=args.ema_inv_gamma,
                  power=args.ema_power,
                  min_value=args.ema_min_value,
                  max_value=args.ema_max_value)
    
    for epoch in tqdm(range(args.epochs)):
        for i, batch in enumerate(dataloader):
            # Move all specific inner dictionary keys to the same device as the model
            for k, v in batch.items():
                batch[k] = v.to(device)
            
            z, loss = policy(batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # use ema model to smoothly update the weights
            if args.use_ema_model:
                ema.step(policy)

        # Save checkpoint 
        if epoch % 100 == 0:
            ckpt = {'policy': policy.state_dict(),
                    'optimizer': optimizer.state_dict()}
            if args.use_ema_model:
                ckpt['ema_policy'] = ema.ema_model.state_dict()
            torch.save(ckpt, os.path.join(saveloc, f'ckpt_{epoch}.pt'))
    
    # Save final model
    ckpt = {'policy': policy.state_dict(),
            'optimizer': optimizer.state_dict()}
    if args.use_ema_model:
        ckpt['ema_policy'] = ema.ema_model.state_dict()
    torch.save(ckpt, os.path.join(saveloc, 'best_model.pt'))


if __name__ == '__main__':
    args = get_config()
    
    # impose random initialization constraints for repeatability 
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    train(args)

    print('='*25)
    print('Done')
    print('='*25)