import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
from collections import deque
from gym_strategy.core.Unit import Soldier

class StrategyEnvDuel(gym.Env):
    """
    Entorno 5x5: 1v1 duelo para capturar o eliminar.
    Acción Discrete(9): moverse o atacar en ortogonales.
    """
    def __init__(self):
        super().__init__()
        self.board_size = (5, 5)
        self.max_turns = 100
        self.capture_turns_required = 3

        # Acciones: 0 quieto, 1 arriba, 2 abajo, 3 izquierda, 4 derecha, 5-8 atacar ortogonal
        self.action_space = spaces.Discrete(9)
        self.observation_space = spaces.Box(0.0, 1.0, shape=(4, *self.board_size), dtype=np.float32)

        self.reset()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.turn_count = 0
        self.current_player = 0  # 0: azul, 1: rojo
        self.capture_progress = [0, 0]  # progreso de captura por equipo

        all_cells = [(x, y) for x in range(5) for y in range(5)]
        while True:
            num_blocks = random.randint(5, 8)
            self.blocked_positions = set(random.sample(all_cells, num_blocks))
            free = [c for c in all_cells if c not in self.blocked_positions]
            if len(free) < 3:
                continue
            spawn1, spawn2, point = random.sample(free, 3)
            
            # 🔵 FORZAR separación mínima entre spawn1 y spawn2
            if (self._manhattan_distance(spawn1, spawn2) >= 6 and
                self._has_path(spawn1, point) and self._has_path(spawn2, point)):
                self.spawn1 = spawn1
                self.spawn2 = spawn2
                self.capture_point = point
                break

        self.units = [
            Soldier(position=self.spawn1, team=0),  # azul
            Soldier(position=self.spawn2, team=1)   # rojo
        ]

        obs = self._get_obs()
        info = {"action_mask": self._get_action_mask()}
        return obs, info

    def step(self, action):
        me = self.units[self.current_player]
        other = self.units[1 - self.current_player]

        dirs = [(0, 0), (0, -1), (0, 1), (-1, 0), (1, 0)]
        attacks = [(0, -1), (0, 1), (-1, 0), (1, 0)]

        reward = -0.01  # Penalización base por usar turno

        if action <= 4:
            # Movimiento
            dx, dy = dirs[action]
            new_pos = (me.position[0] + dx, me.position[1] + dy)
            if self._valid_move(new_pos) and not self._position_occupied(new_pos):
                me.move(new_pos)
        else:
            # Ataque
            attack_dir = attacks[action - 5]
            target_pos = (me.position[0] + attack_dir[0], me.position[1] + attack_dir[1])
            if other.is_alive() and other.position == target_pos:
                other.health -= 50
                reward += 0.1
            else:
                reward -= 0.05

        terminated = False

        # Eliminar enemigo
        if not other.is_alive():
            reward = 1.0
            terminated = True

        # Capturar punto
        if me.position == self.capture_point:
            self.capture_progress[self.current_player] += 1
            if self.capture_progress[self.current_player] >= self.capture_turns_required:
                reward = 1.0
                terminated = True
        else:
            self.capture_progress[self.current_player] = 0

        # Turnos máximos
        self.turn_count += 1
        if self.turn_count >= self.max_turns:
            terminated = True

        # Cambiar turno
        if not terminated:
            self.current_player = 1 - self.current_player

        truncated = False
        obs = self._get_obs()
        info = {"action_mask": self._get_action_mask()}

        if terminated:
            info["episode"] = {"r": reward, "l": self.turn_count}

        return obs, reward, terminated, truncated, info

    def _get_obs(self):
        board = np.zeros((4, *self.board_size), dtype=np.float32)
        for x, y in self.blocked_positions:
            board[0, x, y] = 1.0
        if self.units[0].is_alive():
            ux, uy = self.units[0].position
            board[1, ux, uy] = self.units[0].health / 100
        if self.units[1].is_alive():
            ux, uy = self.units[1].position
            board[2, ux, uy] = self.units[1].health / 100
        cx, cy = self.capture_point
        board[3, cx, cy] = 1.0
        return board

    def _get_action_mask(self):
        mask = np.zeros(9, dtype=np.int8)
        me = self.units[self.current_player]
        other = self.units[1 - self.current_player]
        dirs = [(0, 0), (0, -1), (0, 1), (-1, 0), (1, 0)]

        for a, (dx, dy) in enumerate(dirs):
            nx, ny = me.position[0] + dx, me.position[1] + dy
            if 0 <= nx < 5 and 0 <= ny < 5 and (nx, ny) not in self.blocked_positions:
                if not self._position_occupied((nx, ny)) or (dx, dy) == (0, 0):
                    mask[a] = 1

        attacks = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        for a, (dx, dy) in enumerate(attacks):
            tx, ty = me.position[0] + dx, me.position[1] + dy
            if (0 <= tx < 5 and 0 <= ty < 5) and other.is_alive() and (other.position == (tx, ty)):
                mask[5 + a] = 1

        return mask

    def _valid_move(self, pos):
        x, y = pos
        if not (0 <= x < 5 and 0 <= y < 5):
            return False
        if pos in self.blocked_positions:
            return False
        return True

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
