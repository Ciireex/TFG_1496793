import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
from gym_strategy.core.Unit import Soldier
from collections import deque

class StrategyEnv1vs1(gym.Env):
    def __init__(self):
        super().__init__()
        self.board_size = (5, 5)
        self.max_turns = 50
        self.block_ratio = 0.3  # MÁS obstáculos
        self.action_space = spaces.MultiDiscrete([5, 5])
        self.observation_space = spaces.Box(low=0, high=1, shape=(31,), dtype=np.float32)
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.turn = random.randint(0, 1)
        self.turn_count = 0
        self.done = False
        self.blocked_positions = set()
        self.units = []

        # Generar mapa con camino válido
        while True:
            self.blocked_positions = self._generate_blocked()
            spawn_positions = self._get_spawn_positions()
            if spawn_positions:
                break

        # Crear unidades
        for team, pos in enumerate(spawn_positions):
            self.units.append(Soldier(position=pos, team=team))

        self.episode_reward = 0
        return self._get_obs(), {}

    def _generate_blocked(self):
        all_positions = [(x, y) for x in range(self.board_size[0]) for y in range(self.board_size[1])]
        num_blocks = int(self.block_ratio * len(all_positions))
        return set(random.sample(all_positions, num_blocks))

    def _get_spawn_positions(self):
        empty = [(x, y) for x in range(self.board_size[0]) for y in range(self.board_size[1])
                 if (x, y) not in self.blocked_positions]
        if len(empty) < 2:
            return None
        pos1, pos2 = random.sample(empty, 2)
        if self._has_path(pos1, pos2):
            return [pos1, pos2]
        return None

    def _has_path(self, start, goal):
        visited = set()
        queue = deque([start])
        while queue:
            x, y = queue.popleft()
            if (x, y) == goal:
                return True
            for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
                nx, ny = x + dx, y + dy
                if (0 <= nx < self.board_size[0] and 0 <= ny < self.board_size[1] and
                   (nx, ny) not in self.blocked_positions and (nx, ny) not in visited):
                    visited.add((nx, ny))
                    queue.append((nx, ny))
        return False

    def _get_obs(self):
        me = self.units[self.turn]
        enemy = self.units[1 - self.turn]
        obs = [
            me.position[0] / 4,
            me.position[1] / 4,
            me.health / 100,
            enemy.position[0] / 4,
            enemy.position[1] / 4,
            enemy.health / 100,
        ]
        flat_blocks = [
            1.0 if (x, y) in self.blocked_positions else 0.0
            for y in range(self.board_size[1])
            for x in range(self.board_size[0])
        ]
        return np.array(obs + flat_blocks, dtype=np.float32)

    def step(self, action):
        move, attack = action
        self.turn_count += 1
        reward = 0
        me = self.units[self.turn]
        enemy = self.units[1 - self.turn]

        # Distancia previa
        prev_dist = abs(me.position[0] - enemy.position[0]) + abs(me.position[1] - enemy.position[1])

        # Movimiento
        move_dirs = [(0,0), (0,-1), (0,1), (-1,0), (1,0)]
        dx, dy = move_dirs[move]
        new_pos = (me.position[0] + dx, me.position[1] + dy)
        if self._valid_move(new_pos):
            me.move(new_pos)

        # Distancia nueva y recompensa por acercarse
        new_dist = abs(me.position[0] - enemy.position[0]) + abs(me.position[1] - enemy.position[1])
        if new_dist < prev_dist:
            reward += 0.1
        elif new_dist > prev_dist:
            reward -= 0.05

        # Ataque
        dx, dy = move_dirs[attack]
        target_pos = (me.position[0] + dx, me.position[1] + dy)
        if enemy.position == target_pos:
            enemy.health -= 34
            reward += 0.5  # recompensa por daño
            if enemy.health <= 0:
                reward += 1  # recompensa extra por matar
        else:
            if attack != 0:
                reward -= 0.05  # penaliza ataque al aire

        # Fin de episodio
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

        self.turn = 1 - self.turn
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