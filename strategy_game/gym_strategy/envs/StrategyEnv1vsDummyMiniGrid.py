import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
from gym_strategy.core.Unit import Soldier
from collections import deque

class StrategyEnv1vsDummyMiniGrid(gym.Env):
    def __init__(self, mode='fixed'):
        super().__init__()
        self.board_size = (5, 5)
        self.max_turns = 50
        self.unit_type_map = {"Soldier": 1}
        self.action_space = spaces.MultiDiscrete([5, 5])
        self.observation_space = spaces.Dict({
            "obs": spaces.Box(low=0, high=1, shape=(3, 5, 5), dtype=np.float32)
        })
        self.mode = mode
        self.preset_maps = [
            {(1, 1), (2, 1), (3, 1), (1, 2), (3, 2), (1, 3), (2, 3), (3, 3)},
            {(1, 1), (2, 1), (3, 1), (1, 2), (2, 2), (1, 3), (2, 3), (3, 3)},
            {(1, 1), (2, 1), (3, 1), (2, 2), (3, 2), (1, 3), (3, 3)}
        ]
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.turn = 0
        self.turn_count = 0
        self.done = False
        self.units = []
        self.min_dist = None
        self.last_hit_success = False
        self.no_progress_steps = 0
        self.last_positions = deque(maxlen=3)

        if self.mode == 'fixed':
            self.blocked_positions = {
                (1, 1), (2, 1), (3, 1),
                (1, 2),         (3, 2),
                (1, 3), (2, 3), (3, 3),
            }
        elif self.mode == 'preset':
            self.blocked_positions = random.choice(self.preset_maps)
        elif self.mode == 'random':
            all_positions = [(x, y) for x in range(self.board_size[0]) for y in range(self.board_size[1])]
            num_blocks = random.randint(5, 10)
            self.blocked_positions = set(random.sample(all_positions, num_blocks))
        else:
            raise ValueError("Invalid mode. Use 'fixed', 'preset', or 'random'.")

        self.units.append(Soldier(position=(0, 2), team=0))
        self.units.append(Soldier(position=(4, 2), team=1))

        return self._get_obs(), {}

    def _get_obs(self):
        board = np.zeros((3, self.board_size[0], self.board_size[1]), dtype=np.float32)
        for pos in self.blocked_positions:
            board[0, pos[0], pos[1]] = 1.0
        for unit in self.units:
            if unit.team == self.turn:
                board[1, unit.position[0], unit.position[1]] = unit.health / 100
            else:
                board[2, unit.position[0], unit.position[1]] = unit.health / 100
        return {"obs": board}

    def step(self, action):
        move, attack = action
        self.turn_count += 1
        reward = 0
        me = self.units[self.turn]
        enemy = self.units[1 - self.turn]

        move_dirs = [(0, 0), (0, -1), (0, 1), (-1, 0), (1, 0)]
        dx, dy = move_dirs[move]
        new_pos = (me.position[0] + dx, me.position[1] + dy)
        moved = False
        if self._valid_move(new_pos):
            me.move(new_pos)
            moved = True

        dx, dy = move_dirs[attack]
        target_pos = (me.position[0] + dx, me.position[1] + dy)
        hit = False
        if enemy.position == target_pos:
            enemy.health -= 34
            reward += 0.5
            hit = True
            if enemy.health <= 0:
                reward += 3.0
        elif attack != 0:
            reward -= 0.05

        if self.last_hit_success and not hit:
            reward -= 0.2
        self.last_hit_success = hit

        dist = abs(me.position[0] - enemy.position[0]) + abs(me.position[1] - enemy.position[1])
        if self.min_dist is None:
            self.min_dist = dist
        elif dist < self.min_dist:
            reward += 0.5
            self.min_dist = dist
            self.no_progress_steps = 0
        else:
            self.no_progress_steps += 1

        if self.no_progress_steps >= 5:
            reward -= 0.5
            self.no_progress_steps = 0

        self.last_positions.append(me.position)
        if len(self.last_positions) == 3 and len(set(self.last_positions)) == 1:
            reward -= 0.3  # penaliza por bucle de posición

        done = False
        if enemy.health <= 0:
            reward += 1
            done = True
        elif me.health <= 0:
            reward -= 1
            done = True
        elif self.turn_count >= self.max_turns:
            done = True

        info = {}
        if done:
            info["episode"] = {"r": reward, "l": self.turn_count}

        return self._get_obs(), reward, done, False, info

    def _valid_move(self, pos):
        x, y = pos
        if not (0 <= x < self.board_size[0] and 0 <= y < self.board_size[1]):
            return False
        if pos in self.blocked_positions:
            return False
        if any(u.position == pos for u in self.units):
            return False
        return True
