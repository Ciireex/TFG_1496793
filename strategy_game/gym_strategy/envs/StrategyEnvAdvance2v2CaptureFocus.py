import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
from collections import deque
from gym_strategy.core.Unit import Soldier, Archer

class StrategyEnvAdvance2v2CaptureFocus(gym.Env):
    """
    Mapa 12x5: Azul (PPO) vs Rojo (heurística).
    Entorno optimizado para que el agente aprenda a capturar y defender.
    """
    def __init__(self):
        super().__init__()
        self.board_size = (12, 5)
        self.max_turns = 100
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
            num_blocks = random.randint(5, 8)
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
        reward = 0.0
        terminated = False

        for idx in range(2):
            unit = self.units[idx]
            if not unit.is_alive():
                continue
            a = int(actions[idx])
            self._apply_action(unit, a)

        for idx in range(2, 4):
            unit = self.units[idx]
            if not unit.is_alive():
                continue
            a = self._heuristic_action(unit)
            self._apply_action(unit, a)

        for team in [0, 1]:
            units = self.units[0:2] if team == 0 else self.units[2:4]
            if any(u.is_alive() and u.position == self.capture_point for u in units):
                self.capture_progress[team] += 1
                if team == 0:
                    reward += 0.5
                else:
                    reward -= 0.5
                if self.capture_progress[team] >= self.capture_turns_required:
                    terminated = True
                    reward = 5.0 if team == 0 else -5.0
            else:
                if self.capture_progress[team] > 0:
                    self.capture_progress[team] = 0
                    if team == 0:
                        reward -= 1.0
                    else:
                        reward += 1.0

        if not any(u.is_alive() and u.position == self.capture_point for u in self.units[0:2]):
            reward -= 0.2

        self.turn_count += 1
        if self.turn_count >= self.max_turns:
            terminated = True

        obs = self._get_obs()
        info = {"action_mask": self._get_action_mask()}
        if terminated:
            info["episode"] = {"r": reward, "l": self.turn_count}

        return obs, reward, terminated, False, info

    def _apply_action(self, unit, action):
        dirs = [(0, 0), (0, -1), (0, 1), (-1, 0), (1, 0)]
        attacks = [(0, -1), (0, 1), (-1, 0), (1, 0)]

        if action <= 4:
            dx, dy = dirs[action]
            new_pos = (unit.position[0] + dx, unit.position[1] + dy)
            if self._valid_move(new_pos):
                unit.move(new_pos)
        else:
            dx, dy = attacks[action - 5]
            target_pos = (unit.position[0] + dx, unit.position[1] + dy)
            target = self._get_unit_at(target_pos)
            if target and target.is_alive() and target.team != unit.team:
                unit.attack(target)

    def _heuristic_action(self, unit):
        if unit.position == self.capture_point:
            return 0
        dist_x = self.capture_point[0] - unit.position[0]
        dist_y = self.capture_point[1] - unit.position[1]
        if abs(dist_x) > abs(dist_y):
            return 4 if dist_x > 0 else 3
        else:
            return 2 if dist_y > 0 else 1

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
        masks = []
        dirs = [(0, 0), (0, -1), (0, 1), (-1, 0), (1, 0)]
        for i in range(2):
            unit = self.units[i]
            m = np.zeros(9, dtype=np.int8)
            for a, (dx, dy) in enumerate(dirs):
                np_ = (unit.position[0] + dx, unit.position[1] + dy)
                if self._valid_move(np_):
                    m[a] = 1
            m[5:9] = 1  # Siempre permitir atacar
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
