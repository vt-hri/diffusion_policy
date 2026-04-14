# diffusion_policy
A simple implementaiton of diffusion policy for imitation learning

Dylan Losey, Virginia Tech.

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

## Assignment

This is a minimal implementation of an MLP and a diffusion policy for imitation learning. This is not part of your assignments, but additional reading material to familiarize 
you to state-of-the-art architectures used in imitation learning. To read about the theory of diffusion policy, you can refer to this paper: https://diffusion-policy.cs.columbia.edu/
