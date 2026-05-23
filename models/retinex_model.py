import torch
import torch.nn as nn

class RetinexNet(nn.Module):

    def __init__(self):
        super(RetinexNet, self).__init__()

        # Decomposition: splits input into Reflectance (3ch) + Illumination (1ch)
        self.decomposition = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 4, 3, padding=1),
            nn.Sigmoid()   # Fix: clamp all 4 channels to [0,1]
        )

        # Enhancement: boosts the illumination map
        self.enhancement = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, 3, padding=1),
            nn.Sigmoid()
        )

        # Brightness bias: learned offset to push illumination upward
        self.brightness_bias = nn.Parameter(torch.ones(1) * 0.5)

    def forward(self, x):
        decomposed = self.decomposition(x)

        R = decomposed[:, 0:3, :, :]   # Reflectance: color/detail
        I = decomposed[:, 3:4, :, :]   # Illumination: lighting

        I_enhanced = self.enhancement(I) + self.brightness_bias.clamp(0.0, 1.0)

        output = R * I_enhanced

        # Clamp final output to [0,1] to prevent pixel overflow
        return torch.clamp(output, 0.0, 1.0)