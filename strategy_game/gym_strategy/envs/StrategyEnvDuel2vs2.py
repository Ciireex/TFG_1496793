import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
from collections import deque
from gym_strategy.core.Unit import Soldier, Archer

class StrategyEnvDuel2v2(gym.Env):
    """
    Entorno 8x6: 2v2 (Soldier + Archer por equipo).
    Modo "capture_only" o "full" para entrenar por fases.
    """
    def __init__(self, training_mode="full"):
        super().__init__()
        self.board_size = (8, 6)
        self.max_turns = 80
        self.capture_turns_required = 3
        self.training_mode = training_mode  # "capture_only" o "full"

        self.action_space = spaces.MultiDiscrete([9, 9])
        self.observation_space = spaces.Box(0.0, 1.0, shape=(8, *self.board_size), dtype=np.float32)

        self.reset()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.turn_count = 0
        self.capture_progress = [0, 0]

        all_cells = [(x, y) for x in range(self.board_size[0]) for y in range(self.board_size[1])]
        while True:
            num_blocks = random.randint(6, 10)
            self.blocked_positions = set(random.sample(all_cells, num_blocks))
            free = [c for c in all_cells if c not in self.blocked_positions]
            if len(free) < 5:
                continue
            spawn1, spawn2, spawn3, spawn4, point = random.sample(free, 5)
            if (self._manhattan_distance(spawn1, point) >= 3 and
                self._manhattan_distance(spawn2, point) >= 3 and
                self._manhattan_distance(spawn3, point) >= 3 and
                self._manhattan_distance(spawn4, point) >= 3):
                self.spawns = [spawn1, spawn2, spawn3, spawn4]
                self.capture_point = point
                break

        # Siempre 2 unidades por equipo
        self.units = [
            Soldier(position=self.spawns[0], team=0),
            Archer(position=self.spawns[1], team=0),
            Soldier(position=self.spawns[2], team=1),
            Archer(position=self.spawns[3], team=1),
        ]

        obs = self._get_obs()
        info = {"action_mask": self._get_action_mask()}
        return obs, info

    def step(self, actions):
        reward = 0.0
        terminated = False

        # Azul primero (0 y 1), luego Rojo (2 y 3)
        for i in range(2):
            unit_idx = i if self.turn_count % 2 == 0 else i + 2
            unit = self.units[unit_idx]

            if not unit.is_alive():
                continue

            action = actions[i]
            dirs = [(0, 0), (0, -1), (0, 1), (-1, 0), (1, 0)]
            attacks = [(0, -1), (0, 1), (-1, 0), (1, 0)]

            if action <= 4:
                # Movimiento
                dx, dy = dirs[action]
                new_pos = (unit.position[0] + dx, unit.position[1] + dy)
                if self._valid_move(new_pos) and not self._position_occupied(new_pos):
                    unit.move(new_pos)
            else:
                # Ataque
                if self.training_mode == "full":
                    dx, dy = attacks[action - 5]
                    target_pos = (unit.position[0] + dx, unit.position[1] + dy)
                    target = self._get_unit_at(target_pos)
                    if target and target.is_alive():
                        distance = self._manhattan_distance(unit.position, target.position)
                        if unit.unit_type == "Soldier" and distance == 1:
                            unit.attack(target)
                            reward += 0.1
                        elif unit.unit_type == "Archer" and 1 <= distance <= 3:
                            unit.attack(target)
                            reward += 0.1
                        else:
                            reward -= 0.05
                    else:
                        reward -= 0.05
                else:
                    # En modo capture_only, ignoramos ataque o penalizamos
                    reward -= 0.02

        # Condición de victoria por eliminación solo en "full"
        if self.training_mode == "full":
            team_0_alive = any(u.is_alive() for u in self.units[:2])
            team_1_alive = any(u.is_alive() for u in self.units[2:])

            if not team_0_alive or not team_1_alive:
                terminated = True
                reward = 1.0 if team_0_alive else -1.0

        # Captura del punto
        if not terminated:
            team = 0 if self.turn_count % 2 == 0 else 1
            units_team = self.units[0:2] if team == 0 else self.units[2:4]

            if any(u.position == self.capture_point for u in units_team if u.is_alive()):
                self.capture_progress[team] += 1
                reward += 0.3
                if self.capture_progress[team] >= self.capture_turns_required:
                    terminated = True
                    reward = 1.5
            else:
                if self.capture_progress[team] > 0:
                    reward -= 0.2
                self.capture_progress[team] = 0

        self.turn_count += 1
        if self.turn_count >= self.max_turns:
            terminated = True

        truncated = False
        obs = self._get_obs()
        info = {"action_mask": self._get_action_mask()}

        if terminated:
            info["episode"] = {"r": reward, "l": self.turn_count}

        return obs, reward, terminated, truncated, info

    def _get_obs(self):
        board = np.zeros((8, *self.board_size), dtype=np.float32)
        for x, y in self.blocked_positions:
            board[0, x, y] = 1.0

        if self.units[0].is_alive():
            ux, uy = self.units[0].position
            board[1, ux, uy] = self.units[0].health / 100
        if self.units[1].is_alive():
            ux, uy = self.units[1].position
            board[2, ux, uy] = self.units[1].health / 100
        if self.units[2].is_alive():
            ux, uy = self.units[2].position
            board[3, ux, uy] = self.units[2].health / 100
        if self.units[3].is_alive():
            ux, uy = self.units[3].position
            board[4, ux, uy] = self.units[3].health / 100

        cx, cy = self.capture_point
        board[5, cx, cy] = 1.0

        return board

    def _get_action_mask(self):
        return np.ones((2, 9), dtype=np.int8)

    def _valid_move(self, pos):
        x, y = pos
        return (0 <= x < self.board_size[0] and 0 <= y < self.board_size[1]
                and pos not in self.blocked_positions
                and not self._position_occupied(pos))

    def _position_occupied(self, pos):
        return any(u.is_alive() and u.position == pos for u in self.units)

    def _get_unit_at(self, pos):
        for u in self.units:
            if u.is_alive() and u.position == pos:
                return u
        return None

    def _manhattan_distance(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def _has_path(self, start, goal):
        visited = {start}
        queue = deque([start])
        while queue:
            x, y = queue.popleft()
            if (x, y) == goal:
                return True
            for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
                nx, ny = x + dx, y + dy
                if (0 <= nx < self.board_size[0] and 0 <= ny < self.board_size[1]
                        and (nx, ny) not in self.blocked_positions
                        and (nx, ny) not in visited):
                    visited.add((nx, ny))
                    queue.append((nx, ny))
        return False
