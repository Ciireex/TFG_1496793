
import numpy as np
from collections import deque

class HeuristicPolicy:
    def __init__(self, env):
        self.board_size = env.board_size
        self.capture_point = env.capture_point

    def get_action(self, obs):
        unit_indices = np.argwhere(obs[13] == 1)
        if len(unit_indices) == 0:
            return 0

        unit_pos = tuple(unit_indices[0])
        ux, uy = unit_pos
        is_attack_phase = obs[8, 0, 0] == 1
        team_id = int(obs[9, 0, 0])
        cx, cy = self.capture_point

        ally_layer = 1
        enemy_layer = 4

        equipo_str = "Azul" if team_id == 0 else "Rojo"

        # --- ATAQUE ---
        if is_attack_phase:
            unit_type = self._get_unit_type(obs, unit_pos)
            attack_range = 3 if unit_type == "Archer" else 1
            dirs = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]
            for i, (dx, dy) in enumerate(dirs[1:], start=1):
                for dist in range(1, attack_range + 1):
                    tx, ty = ux + dx * dist, uy + dy * dist
                    if not self._valid_coord((tx, ty)):
                        break
                    if obs[enemy_layer, tx, ty] > 0.5:
                        return i
            return 0

        # --- MOVIMIENTO ---
        if unit_pos == self.capture_point:
            return 0

        if obs[enemy_layer, cx, cy] > 0.5:
            target = (cx, cy)
        elif obs[ally_layer, cx, cy] > 0.5:
            enemy_positions = list(zip(*np.where(obs[enemy_layer] > 0.5)))
            target = self._find_closest(unit_pos, enemy_positions, obs)
        else:
            target = (cx, cy)

        path = self._bfs(unit_pos, target, obs)
        if len(path) >= 2:
            next_pos = path[1]
            dx, dy = next_pos[0] - ux, next_pos[1] - uy
            if dx == -1: return 1
            if dx == 1:  return 2
            if dy == -1: return 3
            if dy == 1:  return 4
        return 0

    def _get_unit_type(self, obs, pos):
        x, y = pos
        if obs[16, x, y] > 0.5: return "Soldier"
        if obs[17, x, y] > 0.5: return "Knight"
        if obs[18, x, y] > 0.5: return "Archer"
        return "Unknown"

    def _bfs(self, start, goal, obs):
        width, height = self.board_size
        visited = set()
        queue = deque([(start, [start])])

        def is_valid(pos):
            x, y = pos
            if pos == goal:
                return 0 <= x < width and 0 <= y < height and pos not in visited
            return (0 <= x < width and 0 <= y < height and
                    obs[0, x, y] == 0 and
                    obs[1, x, y] == 0 and obs[4, x, y] == 0 and
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
