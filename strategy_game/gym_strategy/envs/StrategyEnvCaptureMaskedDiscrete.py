import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
from collections import deque
from gym_strategy.core.Unit import Soldier

class StrategyEnvCaptureMaskedDiscrete(gym.Env):
    """
    Entorno 5×5: un solo soldado debe capturar un punto, moviéndose correctamente.
    Premia acercarse, premia explorar casillas nuevas, penaliza quedarse quieto sin capturar.
    Spawn y captura separados al menos 4 pasos.
    """
    def __init__(self):
        super().__init__()
        self.board_size = (5, 5)
        self.max_turns = 50
        self.capture_turns_required = 3

        self.action_space = spaces.Discrete(5)  # 0: quieto, 1: arriba, 2: abajo, 3: izq, 4: der
        self.observation_space = spaces.Box(0.0, 1.0, shape=(3, *self.board_size), dtype=np.float32)

        self.reset()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.turn_count = 0
        self.capture_progress = 0

        all_cells = [(x, y) for x in range(5) for y in range(5)]
        while True:
            num_blocks = random.randint(5, 10)
            self.blocked_positions = set(random.sample(all_cells, num_blocks))
            free = [c for c in all_cells if c not in self.blocked_positions]
            if len(free) < 2:
                continue
            spawn, point = random.sample(free, 2)

            # 🔵 Separar mínimo 4 pasos
            if self._manhattan_distance(spawn, point) < 4:
                continue

            if self._has_path(spawn, point):
                self.spawn = spawn
                self.capture_point = point
                break

        self.units = [Soldier(position=self.spawn, team=0)]

        # 🔵 Inicializar sistema de exploración
        self.visited_positions = set()
        self.visited_positions.add(self.spawn)
        self.previous_position = self.spawn

        obs = self._get_obs()
        info = {"action_mask": self._get_action_mask()}
        return obs, info

    def step(self, action):
        self.turn_count += 1
        me = self.units[0]
        dirs = [(0, 0), (0, -1), (0, 1), (-1, 0), (1, 0)]
        dx, dy = dirs[int(action)]
        new_pos = (me.position[0] + dx, me.position[1] + dy)

        was_on_capture = (me.position == self.capture_point)
        prev_distance = self._manhattan_distance(me.position, self.capture_point)

        # Mover si válido
        if self._valid_move(new_pos):
            me.move(new_pos)

        now_on_capture = (me.position == self.capture_point)
        new_distance = self._manhattan_distance(me.position, self.capture_point)

        reward = -0.01  # 🔵 Penalización base por turno gastado

        # 🔵 Premio si te acercas
        if new_distance < prev_distance:
            reward += 0.02

        # 🔵 Premio si exploras casilla nueva
        if me.position not in self.visited_positions:
            reward += 0.02
            self.visited_positions.add(me.position)

        # 🔵 Penalización si repites casilla
        if me.position == self.previous_position:
            reward -= 0.02

        self.previous_position = me.position  # Actualizar posición

        # 🔵 Penalizar quieto si no estás capturando
        if action == 0 and not now_on_capture:
            reward -= 0.05

        # 🔵 Lógica de captura
        if now_on_capture:
            if was_on_capture and action == 0:
                reward += 0.05
                self.capture_progress += 1
        else:
            if was_on_capture:
                reward -= 0.2
            self.capture_progress = 0

        # 🔵 Finalizar episodio
        terminated = False
        if self.capture_progress >= self.capture_turns_required:
            reward = 1.0
            terminated = True
        elif self.turn_count >= self.max_turns:
            terminated = True

        truncated = False

        obs = self._get_obs()
        info = {"action_mask": self._get_action_mask()}

        if terminated:
            info["episode"] = {"r": reward, "l": self.turn_count}

        return obs, reward, terminated, truncated, info

    def _get_obs(self):
        board = np.zeros((3, *self.board_size), dtype=np.float32)
        for x, y in self.blocked_positions:
            board[0, x, y] = 1.0
        ux, uy = self.units[0].position
        board[1, ux, uy] = self.units[0].health / 100
        cx, cy = self.capture_point
        board[2, cx, cy] = 1.0
        return board

    def _get_action_mask(self):
        mask = np.zeros(5, dtype=np.int8)
        dirs = [(0, 0), (0, -1), (0, 1), (-1, 0), (1, 0)]
        ux, uy = self.units[0].position
        for a, (dx, dy) in enumerate(dirs):
            nx, ny = ux + dx, uy + dy
            if 0 <= nx < 5 and 0 <= ny < 5 and (nx, ny) not in self.blocked_positions:
                mask[a] = 1
        return mask

    def _valid_move(self, pos):
        x, y = pos
        if not (0 <= x < 5 and 0 <= y < 5):
            return False
        if pos in self.blocked_positions:
            return False
        return True

    def _has_path(self, start, goal):
        visited = {start}
        queue = deque([start])
        while queue:
            x, y = queue.popleft()
            if (x, y) == goal:
                return True
            for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
                nx, ny = x + dx, y + dy
                if (0 <= nx < 5 and 0 <= ny < 5
                        and (nx, ny) not in self.blocked_positions
                        and (nx, ny) not in visited):
                    visited.add((nx, ny))
                    queue.append((nx, ny))
        return False

    def _manhattan_distance(self, pos1, pos2):
        """Calcula la distancia de Manhattan entre dos casillas."""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
