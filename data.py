import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from utils import *


# Standard Dataset class (1 obs -> 1 action)
class Data(Dataset):
    def __init__(self, loadname):
        self.data = pickle.load(open(loadname, "rb"))
        # Unzip the nested tuples into 3 separate collections and convert to stacked FloatTensors dynamically
        # Element format is roughly (static_cam_img, wrist_cam_img, state_action_concated)
        self.static_images, self.ee_images, self.s_a = map(lambda x: torch.FloatTensor(np.stack(x)), zip(*self.data))
        print("imported dataset of length:", len(self.data))

    def __len__(self):
        return len(self.static_images)

    def __getitem__(self,idx):
        out = {'observation': self.s_a[idx, :3],
               'static_image': self.static_images[idx].permute(2, 0, 1),
               'ee_image': self.ee_images[idx].permute(2, 0, 1),
               'action': self.s_a[idx, 3:]}
        return out


# Advanced sequence-based dataloader (seq of obs -> seq of actions)
class DataSequence(Dataset):
    def __init__(self, loadname, obs_horizon, pred_horizon):
        self.obs_horizon = obs_horizon # Length of past sequence input fed to the model
        self.pred_horizon = pred_horizon # Length of sequence to predict ahead
        
        self.data = pickle.load(open(loadname, "rb"))
        self.static_images, self.ee_images, self.s_a = map(lambda x: np.stack(x), zip(*self.data))
        print("imported dataset of length:", len(self.data))
        
        # Hardcoded demonstration length boundary (assumes each task demo consists of exactly 600 frames)
        self.demo_len = 600
        # Calculate maximum possible number of demos
        num_demos = len(self.data) // self.demo_len
        # cumulative array with length of demos to calculate local indexing during sampling
        self.cum_demo_lens = np.arange(1, num_demos + 1) * self.demo_len

    def __len__(self):
        return len(self.static_images)

    def __getitem__(self, index):
        # Finds the specific individual demonstration block corresponding to this continuous rolling flat index 
        demo_idx = np.searchsorted(self.cum_demo_lens, index, side='right')
        # Map the rolling flat index back down into local boundary offsets to correctly sequence internal step bounds 
        local_idx = index if demo_idx == 0 else index - self.cum_demo_lens[demo_idx - 1]

        # Use global fixed length config parameter caching
        demo_len = self.demo_len

        # Clamp observation sequence endpoints
        obs_end = min(local_idx + self.obs_horizon, demo_len)
        act_end = min(local_idx + self.pred_horizon, demo_len)

        # Base slice queries
        obs = self.s_a[local_idx:obs_end, :3]
        actions = self.s_a[local_idx:act_end, 3:]
        static = self.static_images[local_idx:obs_end]
        ego = self.ee_images[local_idx:obs_end]

        # Padding sequences bordering too close to demo bounds
        if obs.shape[0] < self.obs_horizon:
            pad = self.obs_horizon - obs.shape[0]
            # Copy final states
            obs = np.concatenate([obs, np.repeat(obs[-1:], pad, axis=0)])
            static = np.concatenate([static, np.repeat(static[-1:], pad, axis=0)])
            ego = np.concatenate([ego, np.repeat(ego[-1:], pad, axis=0)])
        if actions.shape[0] < self.pred_horizon:
            pad = self.pred_horizon - actions.shape[0]
            actions = np.concatenate([actions, np.repeat(actions[-1:], pad, axis=0)])

        out = {'observation': torch.tensor(obs, dtype=torch.float32),
               'static_image': torch.tensor(static, dtype=torch.float32).permute(0, 3, 1, 2),
               'ee_image': torch.tensor(ego, dtype=torch.float32).permute(0, 3, 1, 2),
               'action': torch.tensor(actions, dtype=torch.float32)}
        return out
