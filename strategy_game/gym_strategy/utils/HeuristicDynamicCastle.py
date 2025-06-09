# HeuristicDynamicCastle.py
import numpy as np
from collections import deque, defaultdict

class HeuristicDynamicCastle:
    def __init__(self, env):
        self.board_size = env.board_size
        self.castle_area = env.castle_area
        self.last_castle_hits = defaultdict(int)

    def get_action(self, obs):
        ux, uy = tuple(np.argwhere(obs[12] == 1)[0])
        unit_id = (ux, uy, int(obs[18, 0, 0]))
        is_attack_phase = obs[17, 0, 0] == 1
        team_id = int(obs[18, 0, 0])
        castle_value = obs[19, 0, 0]

        dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        unit_type = self._get_unit_type(obs, (ux, uy))
        attack_range = 3 if unit_type == "Archer" else 1

        # === ATAQUE ===
        if is_attack_phase:
            for i, (dx, dy) in enumerate(dirs, start=1):
                for d in range(1, attack_range + 1):
                    tx, ty = ux + dx * d, uy + dy * d
                    if not self._valid_coord((tx, ty)):
                        break
                    if obs[7, tx, ty] > 0.5:
                        self.last_castle_hits[unit_id] = 0
                        return i
            for i, (dx, dy) in enumerate(dirs, start=1):
                for d in range(1, attack_range + 1):
                    tx, ty = ux + dx * d, uy + dy * d
                    if not self._valid_coord((tx, ty)):
                        break
                    if obs[1, tx, ty] > 0.5:
                        blocked = any(
                            obs[7, ux + dx * s, uy + dy * s] > 0.5
                            for s in range(1, d)
                            if self._valid_coord((ux + dx * s, uy + dy * s))
                        )
                        if not blocked:
                            self.last_castle_hits[unit_id] += 1
                            return i
            return 0

        # === MOVIMIENTO ===
        goal = None
        if (team_id == 0 and castle_value < 0) or (team_id == 1 and castle_value > 0):
            goal = self._find_castle_approach(obs)
        else:
            if self.last_castle_hits[unit_id] >= 3:
                goal = self._find_closest((ux, uy), list(zip(*np.where(obs[7] > 0.5))), obs)
            else:
                goal = self._find_castle_approach(obs)

        if goal is None:
            goal = self._find_closest((ux, uy), list(zip(*np.where(obs[7] > 0.5))), obs)
        path = self._bfs((ux, uy), goal, obs)

        if len(path) >= 2:
            nx, ny = path[1]
            dx, dy = nx - ux, ny - uy
            if dx == -1: return 1
            if dx == 1: return 2
            if dy == -1: return 3
            if dy == 1: return 4
        return 0

    def _get_unit_type(self, obs, pos):
        x, y = pos
        if obs[13, x, y] > 0.5: return "Soldier"
        if obs[14, x, y] > 0.5: return "Knight"
        if obs[15, x, y] > 0.5: return "Archer"
        return "Unknown"

    def _valid_coord(self, pos):
        x, y = pos
        return 0 <= x < self.board_size[0] and 0 <= y < self.board_size[1]

    def _bfs(self, start, goal, obs):
        from collections import deque
        visited = set()
        queue = deque([(start, [start])])
        def valid(p):
            x, y = p
            return (
                self._valid_coord(p)
                and obs[0, x, y] == 0
                and obs[2, x, y] == 0
                and obs[7, x, y] == 0
                and obs[1, x, y] == 0
                and p not in visited
            )
        while queue:
            (x, y), path = queue.popleft()
            if (x, y) == goal:
                return path
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                np = (x + dx, y + dy)
                if valid(np):
                    visited.add(np)
                    queue.append((np, path + [np]))
        return [start]

    def _find_closest(self, start, targets, obs):
        min_len = float("inf")
        best = None
        for t in targets:
            path = self._bfs(start, t, obs)
            if 1 < len(path) < min_len:
                best = t
                min_len = len(path)
        return best if best else start

    def _find_castle_approach(self, obs):
        for (cx, cy) in self.castle_area:
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
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
