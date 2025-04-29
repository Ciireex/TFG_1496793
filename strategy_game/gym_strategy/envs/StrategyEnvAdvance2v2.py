import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
from collections import deque
from gym_strategy.core.Unit import Soldier, Archer

class StrategyEnvAdvance2v2(gym.Env):
    """
    Estilo Advance Wars: 2v2 Soldier + Archer.
    Mapa horizontal 12x5, con mejoras de aprendizaje.
    """
    def __init__(self):
        super().__init__()
        self.board_size = (12, 5)
        self.max_turns = 100  # 🔵 Límite de turnos bajado
        self.capture_turns_required = 3

        self.action_space = spaces.MultiDiscrete([9, 9])
        self.observation_space = spaces.Box(0.0, 1.0, shape=(6, *self.board_size), dtype=np.float32)

        self.reset()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.turn_count = 0
        self.capture_progress = [0, 0]

        left_side = [(x, y) for x in range(1, 3) for y in range(self.board_size[1])]
        right_side = [(x, y) for x in range(9, 11) for y in range(self.board_size[1])]
        all_cells = [(x, y) for x in range(self.board_size[0]) for y in range(self.board_size[1])]

        while True:
            num_blocks = random.randint(5, 8)  # 🔵 Menos obstáculos
            self.blocked_positions = set(random.sample(all_cells, num_blocks))
            free_left = [c for c in left_side if c not in self.blocked_positions]
            free_right = [c for c in right_side if c not in self.blocked_positions]
            if len(free_left) >= 2 and len(free_right) >= 2:
                spawns_left = random.sample(free_left, 2)
                spawns_right = random.sample(free_right, 2)
                break

        self.spawns = spawns_left + spawns_right

        center_columns = [5, 6]
        center_cells = [(col, row) for col in center_columns for row in range(self.board_size[1])]
        free_center = [c for c in center_cells if c not in self.blocked_positions]
        self.capture_point = random.choice(free_center)

        self.units = [
            Soldier(position=self.spawns[0], team=0),
            Archer(position=self.spawns[1], team=0),
            Soldier(position=self.spawns[2], team=1),
            Archer(position=self.spawns[3], team=1),
        ]

        obs = self._get_obs()
        return obs, {"action_mask": self._get_action_mask()}

    def step(self, actions):
        reward = -0.02  # 🔵 Penalización base si no haces nada útil
        terminated = False

        team = 0 if self.turn_count % 2 == 0 else 1
        base_idx = team * 2

        dirs = [(0, 0), (0, -1), (0, 1), (-1, 0), (1, 0)]
        attacks = [(0, -1), (0, 1), (-1, 0), (1, 0)]

        for i in range(2):
            unit = self.units[base_idx + i]
            if not unit.is_alive():
                continue

            a = int(actions[i])
            prev_dist = self._manhattan_distance(unit.position, self.capture_point)

            if a <= 4:
                dx, dy = dirs[a]
                new_pos = (unit.position[0] + dx, unit.position[1] + dy)
                if self._valid_move(new_pos):
                    unit.move(new_pos)
                new_dist = self._manhattan_distance(unit.position, self.capture_point)
                if new_dist < prev_dist:
                    reward += 0.05
            else:
                dx, dy = attacks[a - 5]
                tx, ty = unit.position[0] + dx, unit.position[1] + dy
                target = self._get_unit_at((tx, ty))
                if target and target.is_alive():
                    dist = abs(dx) + abs(dy)
                    if unit.unit_type == "Soldier" and dist == 1:
                        unit.attack(target)
                        reward += 0.2
                    elif unit.unit_type == "Archer" and 1 <= dist <= 3:
                        unit.attack(target)
                        reward += 0.2
                    else:
                        reward -= 0.05
                else:
                    reward -= 0.05

            if a == 0 and unit.position != self.capture_point:
                reward -= 0.02

        # Captura del punto
        if not terminated:
            group = self.units[0:2] if team == 0 else self.units[2:4]
            if any(u.position == self.capture_point and u.is_alive() for u in group):
                self.capture_progress[team] += 1
                reward += 1.5  # 🔵 Capturar da más puntos
                if self.capture_progress[team] >= self.capture_turns_required:
                    terminated = True
                    reward = 2.0
            else:
                if self.capture_progress[team] > 0:
                    reward -= 0.3
                self.capture_progress[team] = 0

        if self.turn_count >= self.max_turns:
            terminated = True

        self.turn_count += 1

        obs = self._get_obs()
        info = {"action_mask": self._get_action_mask()}
        if terminated:
            info["episode"] = {"r": reward, "l": self.turn_count}

        return obs, reward, terminated, False, info

    def _get_obs(self):
        b = np.zeros((6, *self.board_size), dtype=np.float32)
        for x, y in self.blocked_positions:
            b[0, x, y] = 1.0
        for idx in range(4):
            u = self.units[idx]
            if u.is_alive():
                x, y = u.position
                b[1 + idx, x, y] = u.health / 100
        cx, cy = self.capture_point
        b[5, cx, cy] = 1.0
        return b

    def _get_action_mask(self):
        team = 0 if self.turn_count % 2 == 0 else 1
        base_idx = team * 2
        masks = []

        dirs = [(0, 0), (0, -1), (0, 1), (-1, 0), (1, 0)]
        attacks = [(0, -1), (0, 1), (-1, 0), (1, 0)]

        for i in range(2):
            unit = self.units[base_idx + i]
            m = np.zeros(9, dtype=np.int8)

            for a, (dx, dy) in enumerate(dirs):
                np_ = (unit.position[0] + dx, unit.position[1] + dy)
                if self._valid_move(np_):
                    m[a] = 1

            for a, (dx, dy) in enumerate(attacks, start=5):
                if unit.unit_type == "Soldier":
                    tp = (unit.position[0] + dx, unit.position[1] + dy)
                    tgt = self._get_unit_at(tp)
                    if tgt and tgt.is_alive() and tgt.team != unit.team:
                        m[a] = 1
                elif unit.unit_type == "Archer":
                    for dist in (1, 2, 3):
                        tp = (unit.position[0] + dx * dist, unit.position[1] + dy * dist)
                        tgt = self._get_unit_at(tp)
                        if tgt and tgt.is_alive() and tgt.team != unit.team:
                            m[a] = 1
                            break
            masks.append(m)
        return masks

    def _valid_move(self, pos):
        x, y = pos
        return (0 <= x < self.board_size[0]
                and 0 <= y < self.board_size[1]
                and pos not in self.blocked_positions
                and not any(u.is_alive() and u.position == pos for u in self.units))

    def _get_unit_at(self, pos):
        for u in self.units:
            if u.is_alive() and u.position == pos:
                return u
        return None

    def _manhattan_distance(self, p, q):
        return abs(p[0] - q[0]) + abs(p[1] - q[1])

    def _has_path(self, start, goal):
        visited = {start}
        queue = deque([start])
        while queue:
            x, y = queue.popleft()
            if (x, y) == goal:
                return True
            for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
                nx, ny = x + dx, y + dy
                if (0 <= nx < self.board_size[0]
                   and 0 <= ny < self.board_size[1]
                   and (nx, ny) not in self.blocked_positions
                   and (nx, ny) not in visited):
                    visited.add((nx, ny))
                    queue.append((nx, ny))
        return False
