import numpy as np
from collections import deque

class HeuristicPolicy:
    def __init__(self, env):
        self.board_size = env.board_size
        self.capture_point = env.capture_point

    def get_action(self, obs):
        # Comprobamos si hay unidad activa
        unit_indices = np.argwhere(obs[7] == 1)
        if len(unit_indices) == 0:
            return 0  # Pasar turno si no hay unidad activa

        # Posición de la unidad activa
        unit_pos = tuple(unit_indices[0])
        ux, uy = unit_pos

        # Saber si estamos en fase de ataque
        is_attack_phase = obs[8, 0, 0] == 1

        if not is_attack_phase:
            # --- FASE DE MOVIMIENTO ---

            if unit_pos == self.capture_point:
                return 0

            cx, cy = self.capture_point

            # Si la casilla de captura está vacía → ir a por ella
            if not np.any(obs[1:7, cx, cy]):
                target = (cx, cy)
            else:
                # Si la ocupa un aliado, buscar enemigo más cercano
                enemy_positions = list(zip(*np.where(obs[4, :, :] == 1)))
                target = self._find_closest(unit_pos, enemy_positions, obs)

            path = self._bfs(unit_pos, target, obs)
            if len(path) >= 2:
                next_pos = path[1]
                dx, dy = next_pos[0] - ux, next_pos[1] - uy
                if dx == -1: return 1  # ←
                if dx == 1:  return 2  # →
                if dy == -1: return 3  # ↑
                if dy == 1:  return 4  # ↓

            return 0  # No hay movimiento válido

        else:
            # --- FASE DE ATAQUE ---
            attack_range = 3 if obs[2, ux, uy] == 1.0 else 1
            dirs = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]

            for i, (dx, dy) in enumerate(dirs):
                for dist in range(1, attack_range + 1):
                    tx, ty = ux + dx * dist, uy + dy * dist
                    if not self._valid_coord((tx, ty)):
                        break
                    if obs[4, tx, ty] == 1:
                        return i
            return 0  # No enemigo a rango

    def _bfs(self, start, goal, obs):
        width, height = self.board_size
        visited = set()
        queue = deque([(start, [start])])

        def is_valid(pos):
            x, y = pos
            return (0 <= x < width and 0 <= y < height and
                    obs[0, x, y] == 0 and
                    obs[1, x, y] == 0 and
                    pos not in visited)

        while queue:
            (x, y), path = queue.popleft()
            if (x, y) == goal:
                return path
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dx, y + dy
                new_pos = (nx, ny)
                if is_valid(new_pos):
                    visited.add(new_pos)
                    queue.append((new_pos, path + [new_pos]))
        return [start]

    def _find_closest(self, start, targets, obs):
        min_len = float('inf')
        closest = None
        for pos in targets:
            path = self._bfs(start, pos, obs)
            if len(path) > 1 and len(path) < min_len:
                min_len = len(path)
                closest = pos
        return closest if closest else start

    def _valid_coord(self, pos):
        x, y = pos
        return 0 <= x < self.board_size[0] and 0 <= y < self.board_size[1]
