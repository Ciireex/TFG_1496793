import numpy as np
from collections import deque, defaultdict

class HeuristicCastle:
    def __init__(self, env):
        self.board_size = env.board_size
        self.castle_area = env.castle_area
        self.castle_control = 0
        self.last_castle_hits = defaultdict(int)

    def get_action(self, obs):
        # Identificar unidad activa
        unit_pos = np.argwhere(obs[12] == 1)
        if len(unit_pos) == 0:
            return 0
        ux, uy = tuple(unit_pos[0])
        unit_id = (ux, uy, int(obs[18, 0, 0]))  # pos + equipo

        is_attack_phase = obs[17, 0, 0] == 1
        team_id = int(obs[18, 0, 0])

        if is_attack_phase:
            unit_type = self._get_unit_type(obs, (ux, uy))
            attack_range = 3 if unit_type == "Archer" else 1
            dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]

            # 1. Atacar enemigo
            for i, (dx, dy) in enumerate(dirs, start=1):
                for dist in range(1, attack_range + 1):
                    tx, ty = ux + dx * dist, uy + dy * dist
                    if not self._valid_coord((tx, ty)):
                        break
                    if obs[7, tx, ty] > 0.5:
                        self.last_castle_hits[unit_id] = 0
                        return i

            # 2. Atacar castillo si se puede
            for i, (dx, dy) in enumerate(dirs, start=1):
                for dist in range(1, attack_range + 1):
                    tx, ty = ux + dx * dist, uy + dy * dist
                    if not self._valid_coord((tx, ty)):
                        break
                    if obs[1, tx, ty] > 0.5:
                        blocked = any(
                            obs[7, ux + dx * d, uy + dy * d] > 0.5
                            for d in range(1, dist)
                            if self._valid_coord((ux + dx * d, uy + dy * d))
                        )
                        if not blocked:
                            self.last_castle_hits[unit_id] += 1
                            return i
            return 0

        # Movimiento
        # Si ha atacado 3 veces seguidas al castillo sin progreso, ir a por enemigos
        if self.last_castle_hits[unit_id] >= 3:
            targets = list(zip(*np.where(obs[7] > 0.5)))
            goal = self._find_closest((ux, uy), targets, obs)
        else:
            goal = self._find_castle_approach(obs)
            if goal is None:
                targets = list(zip(*np.where(obs[7] > 0.5)))
                goal = self._find_closest((ux, uy), targets, obs)

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
        if obs[13, x, y] > 0.5:
            return "Soldier"
        elif obs[14, x, y] > 0.5:
            return "Knight"
        elif obs[15, x, y] > 0.5:
            return "Archer"
        else:
            return "Unknown"


    def _valid_coord(self, pos):
        x, y = pos
        return 0 <= x < self.board_size[0] and 0 <= y < self.board_size[1]

    def _bfs(self, start, goal, obs):
        visited = set()
        queue = deque([(start, [start])])

        def is_valid(pos):
            x, y = pos
            if not self._valid_coord(pos): return False
            if pos == goal: return True
            return (
                obs[0, x, y] == 0 and
                obs[2, x, y] == 0 and
                obs[7, x, y] == 0 and
                obs[1, x, y] == 0 and
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
                    obs[1, nx, ny] == 0
                ):
                    return (nx, ny)
        return None
