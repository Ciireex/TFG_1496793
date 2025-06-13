import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class CustomCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=512):
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]

        # Red convolucional con más profundidad y global average pooling
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((1, 1)),  # Hace la red compatible con tableros de cualquier tamaño
            nn.Flatten()
        )

        # Determinar tamaño tras la CNN para crear la capa lineal
        with torch.no_grad():
            sample_input = torch.zeros(1, n_input_channels, *observation_space.shape[1:])
            n_flatten = self.cnn(sample_input).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
            nn.LayerNorm(features_dim)  # Mejor estabilidad que Dropout en RL
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        x = observations.float()
        return self.linear(self.cnn(x))
