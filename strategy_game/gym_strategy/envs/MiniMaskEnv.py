import gymnasium as gym
from gymnasium import spaces
import numpy as np

class MiniMaskEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Dict({
            "obs": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            "action_mask": spaces.Box(low=0, high=1, shape=(5,), dtype=np.int8)
        })
        self.state = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = 0
        obs = self._get_obs()
        return obs, {}

    def step(self, action):
        reward = 1 if action in [0, 1, 2] else -1
        terminated = True
        truncated = False
        info = {}
        obs = self._get_obs()
        return obs, reward, terminated, truncated, info

    def _get_obs(self):
        mask = np.array([1, 1, 1, 0, 0], dtype=np.int8)
        return {
            "obs": np.array([self.state], dtype=np.float32),
            "action_mask": mask
        }
