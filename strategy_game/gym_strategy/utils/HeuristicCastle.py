import numpy as np
from collections import deque

class HeuristicCastle:
    def __init__(self, env):
        self.board_size = env.board_size
        self.castle_area = env.castle_area

    def get_action(self, obs):
        # --- Localizar unidad activa ---
        unit_pos = np.argwhere(obs[12] == 1)
        if len(unit_pos) == 0:
            return 0
        ux, uy = tuple(unit_pos[0])

        # --- Fase de ataque o movimiento ---
        is_attack_phase = obs[17, 0, 0] == 1
        team_id = int(obs[18, 0, 0])

        if is_attack_phase:
            unit_type = self._get_unit_type(obs, (ux, uy))
            attack_range = 3 if unit_type == "Archer" else 1
            dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # ↑ ↓ ← →
            
            # 1. Prioridad: atacar enemigo
            for i, (dx, dy) in enumerate(dirs, start=1):
                for dist in range(1, attack_range + 1):
                    tx, ty = ux + dx * dist, uy + dy * dist
                    if not self._valid_coord((tx, ty)):
                        break
                    if obs[7, tx, ty] > 0.5:
                        return i
            
            # 2. Si no hay enemigos, atacar castillo si está en línea y sin bloqueo
            for i, (dx, dy) in enumerate(dirs, start=1):
                for dist in range(1, attack_range + 1):
                    tx, ty = ux + dx * dist, uy + dy * dist
                    if not self._valid_coord((tx, ty)):
                        break
                    if obs[1, tx, ty] > 0.5:
                        # Asegurarse de que no hay enemigos entre medias
                        blocked = any(
                            obs[7, ux + dx * d, uy + dy * d] > 0.5
                            for d in range(1, dist)
                            if self._valid_coord((ux + dx * d, uy + dy * d))
                        )
                        if not blocked:
                            return i
            return 0

        # --- Movimiento hacia castillo o enemigos ---
        goal = self._find_castle_approach(obs)
        if goal is None:
            enemy_pos = list(zip(*np.where(obs[7] > 0.5)))
            goal = self._find_closest((ux, uy), enemy_pos, obs)
            if goal is None:
                return 0

        path = self._bfs((ux, uy), goal, obs)
        if len(path) >= 2:
            nx, ny = path[1]
            dx, dy = nx - ux, ny - uy
            if dx == -1: return 1
            if dx == 1:  return 2
            if dy == -1: return 3
            if dy == 1:  return 4
        return 0

    def _get_unit_type(self, obs, pos):
        x, y = pos
        if obs[13, x, y] > 0.5: return "Soldier"
        if obs[14, x, y] > 0.5: return "Knight"
        if obs[15, x, y] > 0.5: return "Archer"
        return "Unknown"

    def _bfs(self, start, goal, obs):
        visited = set()
        queue = deque([(start, [start])])
        width, height = self.board_size

        def is_valid(pos):
            x, y = pos
            if not self._valid_coord(pos): return False
            if pos == goal: return True
            return (
                obs[0, x, y] == 0 and  # sin obstáculo
                obs[2, x, y] == 0 and  # sin aliado
                obs[7, x, y] == 0 and  # sin enemigo
                obs[1, x, y] == 0 and  # sin castillo (no se puede pisar)
                pos not in visited
            )

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

    def _find_castle_approach(self, obs):
        for (cx, cy) in self.castle_area:
            for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
                nx, ny = cx + dx, cy + dy
                if not self._valid_coord((nx, ny)): continue
                if (
                    obs[0, nx, ny] == 0 and
                    obs[2, nx, ny] == 0 and
                    obs[7, nx, ny] == 0 and
                    obs[1, nx, ny] == 0  # no encima del castillo
                ):
                    return (nx, ny)
        return None

    def _valid_coord(self, pos):
        x, y = pos
        return 0 <= x < self.board_size[0] and 0 <= y < self.board_size[1]
