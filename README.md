# diffusion_policy
A simple implementaiton of diffusion policy for imitation learning

Sagar Parekh, Virginia Tech.

## Install and Run

```bash

# Download
git clone https://github.com/vt-hri/diffusion_policy.git
cd diffusion_policy

# Create and source virtual environment
python3 -m venv venv
source venv/bin/activate

# if you are using Mac or Conda, use the following lines
conda create -n venv python=3.10 -y
conda activate venv

# Install dependencies (for Linux and WSL)
pip install numpy pybullet torch torchvision tqdm diffusers

# Install dependencies (for Mac)
pip install numpy torch torchvision tqdm diffusers
conda install pybullet
```

## Description

This is a minimal implementation of an MLP and a diffusion policy for imitation learning. This is not part of your assignments, but additional reading material to familiarize 
you to state-of-the-art architectures used in imitation learning. To read about the theory of diffusion policy, you can refer to this paper: https://diffusion-policy.cs.columbia.edu/

## Train

To train a Behavior Cloning policy, run the following command:

```
python train.py --dataset_path ./dataset.pkl --device cuda --policy [policy_type] --savename [save_name] --epochs 500
```

The ```--policy``` argument has two options:
- mlp (to train a multi layer perceptron action head)
- diffusion (to train a diffusion policy)

For diffusion policy, you can also specify the length of input sequence as well as the length of predicted actions using the arguments ```--obs_horizon``` and ```--pred_horizon``` respectively.

The trained models are stored in ```./results/PickAndPlace/[save_name]```.

## Evaluation
To evaluate the trained policy in the simulated environment, run the following command:

```
python eval.py --num_evals 100 --device cuda --loadloc [path_to_trained_model] --n_rollout_actions [n_rollout_actions]
```

The ```--n_rollout_actions``` argument allows you to specify the length of the predicted action sequence that will be executed.

The results are saved in the same directory as the trained models.

