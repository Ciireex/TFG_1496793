import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
from gym_strategy.core.Unit import Soldier
from collections import deque

class StrategyEnvCapturePersist(gym.Env):
    def __init__(self):
        super().__init__()
        self.board_size = (5, 5)
        self.max_turns = 50
        self.capture_turns_required = 3
        self.action_space = spaces.MultiDiscrete([5, 5])
        self.observation_space = spaces.Dict({
            "obs": spaces.Box(low=0, high=1, shape=(3, 5, 5), dtype=np.float32)
        })
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.turn = 0
        self.turn_count = 0
        self.done = False
        self.units = []
        self.capture_progress = 0
        self.last_positions = deque(maxlen=3)
        self.visited = set()

        while True:
            all_positions = [(x, y) for x in range(self.board_size[0]) for y in range(self.board_size[1])]
            self.blocked_positions = set(random.sample(all_positions, random.randint(5, 10)))
            candidates = [p for p in all_positions if p not in self.blocked_positions]
            if len(candidates) < 2:
                continue
            spawn, point = random.sample(candidates, 2)
            if self._has_path(spawn, point):
                self.capture_point = point
                self.units.append(Soldier(position=spawn, team=0))
                break

        return self._get_obs(), {}

    def _get_obs(self):
        board = np.zeros((3, *self.board_size), dtype=np.float32)
        for pos in self.blocked_positions:
            board[0, pos[0], pos[1]] = 1.0
        for unit in self.units:
            board[1, unit.position[0], unit.position[1]] = unit.health / 100
        cx, cy = self.capture_point
        board[2, cx, cy] = 1.0
        return {"obs": board}

    def step(self, action):
        move, _ = action
        self.turn_count += 1
        reward = 0
        me = self.units[0]

        move_dirs = [(0, 0), (0, -1), (0, 1), (-1, 0), (1, 0)]
        dx, dy = move_dirs[move]
        new_pos = (me.position[0] + dx, me.position[1] + dy)

        if self._valid_move(new_pos):
            me.move(new_pos)
            if me.position not in self.visited:
                reward += 0.1
                self.visited.add(me.position)
        else:
            reward -= 0.2

        if me.position == self.capture_point:
            self.capture_progress += 1
            reward += 0.5
            if self.capture_progress >= self.capture_turns_required:
                reward += 3.0
                self.done = True
        else:
            if self.capture_progress > 0:
                reward -= 1.0  # penaliza bajarse del punto
            self.capture_progress = 0

        self.last_positions.append(me.position)
        if len(self.last_positions) == 3 and len(set(self.last_positions)) == 1:
            reward -= 0.3

        done = self.done or self.turn_count >= self.max_turns
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
        return True

    def _has_path(self, start, goal):
        visited = set()
        queue = deque([start])
        while queue:
            x, y = queue.popleft()
            if (x, y) == goal:
                return True
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dx, y + dy
                if (0 <= nx < self.board_size[0] and 0 <= ny < self.board_size[1] and
                        (nx, ny) not in self.blocked_positions and (nx, ny) not in visited):
                    visited.add((nx, ny))
                    queue.append((nx, ny))
        return False
