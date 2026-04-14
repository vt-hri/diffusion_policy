import torch
from torchvision.models import resnet18, ResNet18_Weights
import torch.nn as nn
import numpy as np

from utils import *


# Image encoder for processing image observations using pretrained ResNet architecture
class ImageEncoder(nn.Module):
    def __init__(self, feature_dim=128):
        super(ImageEncoder, self).__init__()

        # Fetch foundational ResNet-18 mappings dynamically downloading weights properly 
        weights = ResNet18_Weights.DEFAULT
        cnn_backbone = resnet18(weights)

        # Discard unstable BatchNormalization operations
        self._replace_bn_with_gn(cnn_backbone)
        # Detach final classification layer of ResNet
        cnn_backbone.fc = nn.Identity()
        enc_list = [cnn_backbone]

        # Insert custom dimensional bottleneck projector  
        linear = nn.Linear(512, feature_dim)
        enc_list.append(linear)
        self.enc = nn.Sequential(*enc_list)

    def forward(self, x):
        # normalize images to range limits 0-1
        x /= 255.
        features = self.enc(x)
        return features
    
    # replace batch normalization layers with group normalization layers (helps with diffusion policies)
    def _replace_bn_with_gn(self, cnn, num_groups=32):
        for name, child in cnn.named_children():
            if isinstance(child, nn.BatchNorm2d):
                num_channels = child.num_features

                g = min(num_groups, num_channels)
                while num_channels % g != 0:
                    g -= 1

                setattr(cnn, name, nn.GroupNorm(num_groups=g,
                                                num_channels=num_channels,
                                                eps=child.eps,
                                                affine=True))
            else:
                self._replace_bn_with_gn(child, num_groups)


# Encoder block that processes all image observations
class Encoder(nn.Module):
    def __init__(self, state_dim):
        super(Encoder, self).__init__()

        self.x_dim = state_dim
        
        self.visual_encoders = nn.ModuleDict()
        # Allocate separate CNN instances handling distinct camera view independently
        self.visual_encoders['static_image'] = ImageEncoder()
        self.visual_encoders['ee_image'] = ImageEncoder()

    def forward(self, batch):
        image_keys = [key for key in batch.keys() if 'image' in key]

        h = []
        for key in image_keys:
            img = batch[key]
            shape = img.shape[:-3]
            if len(shape) > 1:
                img = img.flatten(end_dim=-4)
            x = self.visual_encoders[key](img)
            h.append(x.reshape(*shape, -1))
        h = torch.cat(h, dim=-1)
        features = torch.cat((h, batch['observation']), dim=-1)
        return features
        