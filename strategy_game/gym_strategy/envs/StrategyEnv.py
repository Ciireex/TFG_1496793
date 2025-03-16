import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gym_strategy.core.Board import Board
from gym_strategy.core.Unit import Soldier

class StrategyEnv(gym.Env):
    def __init__(self):
        super(StrategyEnv, self).__init__()
        self.board = Board()
        self.action_space = spaces.Discrete(100)  # Actions: move, attack, capture, etc.
        self.observation_space = spaces.Box(low=0, high=1, shape=(10, 10), dtype=np.float32)
        self.reset()

    def step(self, action):
        reward = 0
        done = False

        # Implement action logic (e.g., move unit, attack)
        # Update rewards accordingly

        return self._get_state(), reward, done, {}

    def reset(self):
        self.board = Board()
        self.board.add_unit(Soldier((0, 0), team=0))
        self.board.add_unit(Soldier((9, 9), team=1))
        return self._get_state()

    def _get_state(self):
        state = np.zeros((10, 10))
        for unit in self.board.units:
            state[unit.position] = 1 if unit.team == 0 else -1
        return state
