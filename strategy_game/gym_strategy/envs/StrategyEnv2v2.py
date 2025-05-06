import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
from collections import deque
from gym_strategy.core.Unit import Soldier

class StrategyEnv2v2(gym.Env):
    """
    Entorno 5x5: 2v2 duelo para capturar o eliminar.
    Acción MultiDiscrete([9, 9]): moverse o atacar en ortogonales para cada unidad.
    """
    def __init__(self):
        super().__init__()
        self.board_size = (5, 5)
        self.max_turns = 100
        self.capture_turns_required = 3

        self.action_space = spaces.MultiDiscrete([9, 9])
        self.observation_space = spaces.Dict({
            "obs": spaces.Box(0.0, 1.0, shape=(6, *self.board_size), dtype=np.float32),
            "action_mask": spaces.MultiBinary((2, 9))
        })

        self.reset()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.turn_count = 0
        self.current_player = 0  # 0: azul, 1: rojo
        self.capture_progress = [0, 0]
        self.previous_positions = {}  # Para castigar repeticiones

        left_edge = [(x, 0) for x in range(5)]
        right_edge = [(x, 4) for x in range(5)]
        center_options = [(x, 2) for x in range(1, 4)]

        while True:
            num_blocks = random.randint(4, 7)
            all_cells = [(x, y) for x in range(5) for y in range(5)]
            self.blocked_positions = set(random.sample(all_cells, num_blocks))

            free = [c for c in all_cells if c not in self.blocked_positions]
            blue_side = left_edge if random.random() < 0.5 else right_edge
            red_side = right_edge if blue_side == left_edge else left_edge

            blue_spawns = [c for c in blue_side if c in free]
            red_spawns = [c for c in red_side if c in free]
            centers = [c for c in center_options if c in free]

            if len(blue_spawns) >= 2 and len(red_spawns) >= 2 and len(centers) >= 1:
                self.spawn_blue = random.sample(blue_spawns, 2)
                self.spawn_red = random.sample(red_spawns, 2)
                self.capture_point = random.choice(centers)
                break

        self.units = [
            Soldier(position=self.spawn_blue[0], team=0),
            Soldier(position=self.spawn_blue[1], team=0),
            Soldier(position=self.spawn_red[0], team=1),
            Soldier(position=self.spawn_red[1], team=1)
        ]

        obs = self._get_obs()
        obs_dict = {"obs": obs, "action_mask": self._get_action_mask()}
        return obs_dict, {}

    def step(self, actions):
        reward = -0.02
        terminated = False

        active_units = [u for u in self.units if u.team == self.current_player and u.is_alive()]
        enemy_units = [u for u in self.units if u.team != self.current_player and u.is_alive()]

        dirs = [(0, 0), (0, -1), (0, 1), (-1, 0), (1, 0)]
        attacks = [(0, -1), (0, 1), (-1, 0), (1, 0)]

        for i, (unit, action) in enumerate(zip(active_units, actions)):
            key = (self.current_player, i)
            previous_pos = self.previous_positions.get(key)

            if action <= 4:
                dx, dy = dirs[action]
                new_pos = (unit.position[0] + dx, unit.position[1] + dy)
                if self._valid_move(new_pos) and not self._position_occupied(new_pos):
                    unit.move(new_pos)
                    if previous_pos == new_pos:
                        reward -= 0.05  # penaliza repetir movimiento sin avanzar
                    self.previous_positions[key] = new_pos
            else:
                dx, dy = attacks[action - 5]
                target = (unit.position[0] + dx, unit.position[1] + dy)
                for enemy in enemy_units:
                    if enemy.is_alive() and enemy.position == target:
                        enemy.health -= 50
                        reward += 0.1
                        break
                else:
                    reward -= 0.1

            # Captura
            if unit.position == self.capture_point:
                self.capture_progress[self.current_player] += 1
                reward += 0.1
                if self.capture_progress[self.current_player] >= self.capture_turns_required:
                    reward = 1.5
                    terminated = True
            else:
                self.capture_progress[self.current_player] = 0

        if not enemy_units:
            reward = 1.5
            terminated = True

        self.turn_count += 1
        if self.turn_count >= self.max_turns:
            terminated = True

        if not terminated:
            self.current_player = 1 - self.current_player

        obs = self._get_obs()
        obs_dict = {"obs": obs, "action_mask": self._get_action_mask()}
        info = {}
        if terminated:
            info["episode"] = {"r": reward, "l": self.turn_count}

        return obs_dict, reward, terminated, False, info

    def _get_obs(self):
        board = np.zeros((6, *self.board_size), dtype=np.float32)
        for x, y in self.blocked_positions:
            board[0, x, y] = 1.0
        for i, unit in enumerate(self.units):
            if unit.is_alive():
                x, y = unit.position
                board[1 + unit.team * 2 + i % 2, x, y] = unit.health / 100
        cx, cy = self.capture_point
        board[5, cx, cy] = 1.0
        return board

    def _get_action_mask(self):
        mask = np.zeros((2, 9), dtype=np.int8)
        dirs = [(0, 0), (0, -1), (0, 1), (-1, 0), (1, 0)]
        attacks = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        my_units = [u for u in self.units if u.team == self.current_player and u.is_alive()]
        enemies = [u for u in self.units if u.team != self.current_player and u.is_alive()]

        for i, unit in enumerate(my_units):
            for a, (dx, dy) in enumerate(dirs):
                nx, ny = unit.position[0] + dx, unit.position[1] + dy
                if 0 <= nx < 5 and 0 <= ny < 5 and (nx, ny) not in self.blocked_positions:
                    if not self._position_occupied((nx, ny)) or (dx, dy) == (0, 0):
                        mask[i, a] = 1
            for a, (dx, dy) in enumerate(attacks):
                tx, ty = unit.position[0] + dx, unit.position[1] + dy
                if 0 <= tx < 5 and 0 <= ty < 5:
                    for e in enemies:
                        if e.position == (tx, ty):
                            mask[i, 5 + a] = 1
                            break
        return mask

    def _valid_move(self, pos):
        x, y = pos
        return 0 <= x < 5 and 0 <= y < 5 and pos not in self.blocked_positions

    def _position_occupied(self, pos):
        return any(u.is_alive() and u.position == pos for u in self.units)

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
                if (0 <= nx < 5 and 0 <= ny < 5 and (nx, ny) not in self.blocked_positions
                        and (nx, ny) not in visited):
                    visited.add((nx, ny))
                    queue.append((nx, ny))
        return False
