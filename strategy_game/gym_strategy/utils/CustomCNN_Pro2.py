import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y_avg = self.fc(self.avg_pool(x).view(b, c)).view(b, c, 1, 1)
        y_max = self.fc(self.max_pool(x).view(b, c)).view(b, c, 1, 1)
        return x * (y_avg + y_max)


class DynamicSpatialAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.attn_conv = nn.Conv2d(channels, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.bn = nn.BatchNorm2d(channels)  # Para mayor estabilidad

    def forward(self, x):
        x = self.bn(x)
        attn_weights = self.sigmoid(self.attn_conv(x))  # [B, 1, H, W]
        return x * attn_weights


class HybridCNN(nn.Module):
    """Backbone con atención espacial y de canal, optimizado para mapas pequeños"""
    def __init__(self, n_input_channels):
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(n_input_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            ChannelAttention(64),
            DynamicSpatialAttention(64),
            nn.Dropout2d(0.05)
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128)
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            ChannelAttention(256),
            DynamicSpatialAttention(256)
        )

        self.pool = nn.AdaptiveAvgPool2d((2, 2))

    def forward(self, x):
        x = self.block1(x.float())
        x = self.block2(x)
        x = self.block3(x)
        return torch.flatten(self.pool(x), start_dim=1)


class EnhancedTacticalFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=384):
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]

        # Usa HybridCNN como backbone principal
        self.cnn = HybridCNN(n_input_channels)

        # Calcula el tamaño de salida del backbone
        with torch.no_grad():
            sample = torch.zeros(1, *observation_space.shape)
            n_flatten = self.cnn(sample).shape[1]

        # Procesamiento final de las features
        self.feature_net = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
            nn.LayerNorm(features_dim)
        )

        # Procesamiento del canal del equipo activo (canal 18)
        self.team_net = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU()
        )

        self._features_dim = features_dim + 32

    def forward(self, x):
        # Extrae features espaciales
        cnn_features = self.cnn(x)
        tactical_features = self.feature_net(cnn_features)

        # Extrae el valor medio del canal del equipo actual (canal 18)
        team_channel = x[:, 18:19, :, :]  # [B, 1, H, W]
        team_scalar = team_channel.view(team_channel.size(0), -1).mean(dim=1, keepdim=True)
        team_scalar = torch.nan_to_num(team_scalar, nan=0.0)

        team_features = self.team_net(team_scalar)

        return torch.cat([tactical_features, team_features], dim=1)
