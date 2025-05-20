import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch.nn as nn
import torch

class SmallMapCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box):
        super().__init__(observation_space, features_dim=64)
        n_input_channels = observation_space.shape[0]

        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )

        # Calcular el tamaño de entrada del Linear
        with torch.no_grad():
            sample = torch.zeros((1, *observation_space.shape))
            n_flatten = self.cnn(sample).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, 64),
            nn.ReLU()
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))
