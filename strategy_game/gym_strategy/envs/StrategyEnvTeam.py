import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
from gym_strategy.core.Unit import Soldier
from collections import deque

class StrategyEnvTeam(gym.Env):
    def __init__(self):
        super().__init__()
        self.board_size = (5, 5)
        self.max_turns = 100
        self.units_per_team = 3
        self.block_ratio = 0.3
        self.action_space = spaces.MultiDiscrete([5, 5])  # mover, atacar
        self.observation_space = spaces.Box(low=0, high=1, shape=(6 + 25,), dtype=np.float32)  # unidad activa + obstaculos
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.turn = random.randint(0, 1)
        self.turn_count = 0
        self.done = False
        self.blocked_positions = set()

        while True:
            self.blocked_positions = self._generate_blocked()
            spawn_positions = self._get_spawn_positions()
            if spawn_positions:
                break

        self.units = []
        for team in [0, 1]:
            for i in range(self.units_per_team):
                self.units.append(Soldier(position=spawn_positions[team][i], team=team))

        self.unit_index = 0  # índice de unidad del equipo actual
        self.episode_reward = 0
        return self._get_obs(), {}

    def _generate_blocked(self):
        all_positions = [(x, y) for x in range(self.board_size[0]) for y in range(self.board_size[1])]
        num_blocks = int(self.block_ratio * len(all_positions))
        return set(random.sample(all_positions, num_blocks))

    def _get_spawn_positions(self):
        empty = [(x, y) for x in range(self.board_size[0]) for y in range(self.board_size[1])
                 if (x, y) not in self.blocked_positions]
        if len(empty) < self.units_per_team * 2:
            return None
        positions = random.sample(empty, self.units_per_team * 2)
        team1 = positions[:self.units_per_team]
        team2 = positions[self.units_per_team:]

        # Validar que todas las unidades de cada equipo tengan camino a las del otro
        for pos1 in team1:
            if not any(self._has_path(pos1, pos2) for pos2 in team2):
                return None
        return [team1, team2]

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
        team_units = [u for u in self.units if u.team == self.turn and u.is_alive()]
        unit = team_units[self.unit_index % len(team_units)]
        enemy_units = [u for u in self.units if u.team != self.turn and u.is_alive()]

        obs = [
            unit.position[0] / 4,
            unit.position[1] / 4,
            unit.health / 100,
            enemy_units[0].position[0] / 4 if enemy_units else 0,
            enemy_units[0].position[1] / 4 if enemy_units else 0,
            enemy_units[0].health / 100 if enemy_units else 0,
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
        done = False

        team_units = [u for u in self.units if u.team == self.turn and u.is_alive()]
        if not team_units:
            self.done = True
            return self._get_obs(), -1, self.done, False, {}

        unit = team_units[self.unit_index % len(team_units)]
        enemy_units = [u for u in self.units if u.team != self.turn and u.is_alive()]

        # Movimiento
        move_dirs = [(0,0), (0,-1), (0,1), (-1,0), (1,0)]
        dx, dy = move_dirs[move]
        new_pos = (unit.position[0] + dx, unit.position[1] + dy)
        if self._valid_move(new_pos):
            unit.move(new_pos)

        # Ataque
        atk_dirs = move_dirs
        dx, dy = atk_dirs[attack]
        target_pos = (unit.position[0] + dx, unit.position[1] + dy)
        for enemy in enemy_units:
            if enemy.position == target_pos:
                enemy.health -= 34
                reward += 0.5  # recompensa positiva por hacer daño
                if enemy.health <= 0:
                    reward += 1  # más recompensa por matar

        # Recompensa negativa si no se hizo nada útil
        if reward == 0 and move == 0 and attack == 0:
            reward -= 0.05

        if all(not u.is_alive() for u in self.units if u.team != self.turn):
            self.done = True
            reward += 3  # victoria total

        if self.turn_count >= self.max_turns:
            self.done = True

        self.unit_index += 1
        if self.unit_index >= len([u for u in self.units if u.team == self.turn and u.is_alive()]):
            self.unit_index = 0
            self.turn = 1 - self.turn

        return self._get_obs(), reward, self.done, False, {}

    def _valid_move(self, pos):
        x, y = pos
        if not (0 <= x < self.board_size[0] and 0 <= y < self.board_size[1]):
            return False
        if pos in self.blocked_positions:
            return False
        if any(u.position == pos and u.is_alive() for u in self.units):
            return False
        return True