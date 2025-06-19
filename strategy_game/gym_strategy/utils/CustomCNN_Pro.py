import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class DynamicSpatialAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.attn_conv = nn.Conv2d(channels, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        attn_weights = self.sigmoid(self.attn_conv(x))
        return x * attn_weights

class CustomCNN_Tactical(nn.Module):
    def __init__(self, n_input_channels, map_size):
        super().__init__()
        self.map_size = map_size
        self.stride = 1 if map_size <= (6, 4) else 2

        self.block1 = nn.Sequential(
            nn.Conv2d(n_input_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout2d(0.1),
            DynamicSpatialAttention(64)
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=self.stride, padding=1),
            nn.ReLU(),
            nn.Dropout2d(0.1)
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            DynamicSpatialAttention(256)
        )

        self.pool = nn.AdaptiveAvgPool2d((2, 2) if map_size <= (6, 4) else (3, 2))

    def forward(self, x):
        x = self.block1(x.float())
        x = self.block2(x)
        x = self.block3(x)
        return torch.flatten(self.pool(x), start_dim=1)

class TacticalFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=512):
        n_input_channels = observation_space.shape[0]
        map_size = observation_space.shape[1:]

        super().__init__(observation_space, features_dim=features_dim)  # se sobrescribe luego

        self.backbone = CustomCNN_Tactical(n_input_channels, map_size)

        with torch.no_grad():
            sample = torch.zeros(1, *observation_space.shape)
            n_flatten = self.backbone(sample).shape[1]

        self.tactical_net = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.SiLU(),
            nn.Linear(features_dim, features_dim),
            nn.LayerNorm(features_dim),
            nn.Dropout(0.3)
        )

        self.terrain_head = nn.Linear(features_dim, 32)
        self.unit_head = nn.Linear(features_dim, 32)

        # 💡 NUEVO: head para procesar el canal de equipo activo (capa 18)
        self.team_head = nn.Sequential(
            nn.Linear(1, 32),
            nn.SiLU()
        )

        # ✅ Total de features: main + terrain + unit + team
        self._features_dim = features_dim + 32 + 32 + 32

    def forward(self, x):
        features = self.backbone(x)
        tactical = self.tactical_net(features)
        terrain_feat = self.terrain_head(tactical)
        unit_feat = self.unit_head(tactical)

        # Extrae canal 18 y reduce a [batch, 1] con el valor del equipo activo
        team_channel = x[:, 18:19, :, :]            # [B, 1, H, W]
        team_scalar = team_channel.mean(dim=(2, 3)) # [B, 1]
        team_feat = self.team_head(team_scalar)     # [B, 32]

        return torch.cat([tactical, terrain_feat, unit_feat, team_feat], dim=1)
