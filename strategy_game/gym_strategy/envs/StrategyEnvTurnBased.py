import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
from collections import deque
from gym_strategy.core.Unit import Soldier, Archer

class StrategyEnvTurnBased(gym.Env):
    def __init__(self, blue_team=None, red_team=None):
        super().__init__()
        self.board_size = (9, 6)
        self.max_turns = 60
        self.capture_turns_required = 3

        self.blue_team = blue_team if blue_team else [Soldier, Soldier, Archer]
        self.red_team = red_team if red_team else [Archer, Soldier, Soldier]
        self.unit_types = self.blue_team + self.red_team
        self.num_units = len(self.unit_types)

        self.action_space = spaces.Discrete(9)  # 5 mov + 4 ataques
        self.observation_space = spaces.Dict({
            "obs": spaces.Box(0.0, 1.0, shape=(10, *self.board_size), dtype=np.float32),
            "action_mask": spaces.MultiBinary(9)
        })

        self.reset()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.turn_count = 0
        self.current_player = 0
        self.active_unit_idx = 0
        self.capture_progress = [0, 0]

        width, height = self.board_size
        while True:
            all_cells = [(x, y) for x in range(width) for y in range(height)]
            self.blocked_positions = set(random.sample(all_cells, random.randint(5, 10)))
            free = [c for c in all_cells if c not in self.blocked_positions]

            blue_side = [(0, y) for y in range(height)]
            red_side = [(width - 1, y) for y in range(height)]
            center_options = [(width // 2, y) for y in range(2, 4)]

            blue_spawns = [c for c in blue_side if c in free]
            red_spawns = [c for c in red_side if c in free]
            centers = [c for c in center_options if c in free]

            if len(blue_spawns) >= 3 and len(red_spawns) >= 3 and centers:
                self.spawn_blue = random.sample(blue_spawns, 3)
                self.spawn_red = random.sample(red_spawns, 3)
                self.capture_point = random.choice(centers)

                if all(self._has_path(s, self.capture_point) for s in self.spawn_blue + self.spawn_red):
                    break

        self.units = []
        for i, unit_cls in enumerate(self.unit_types):
            team = 0 if i < 3 else 1
            spawn = self.spawn_blue[i] if team == 0 else self.spawn_red[i - 3]
            self.units.append(unit_cls(position=spawn, team=team))

        obs = self._get_obs()
        return {"obs": obs, "action_mask": self._get_action_mask()}, {}

    def step(self, action):
        reward = -0.01
        terminated = False

        dirs = [(0, 0), (0, -1), (0, 1), (-1, 0), (1, 0)]
        attacks_melee = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        attacks_archer = [(dx, dy) for dx in range(-3, 4) for dy in range(-3, 4)
                          if abs(dx) + abs(dy) in [2, 3] and (dx == 0 or dy == 0)]

        active_units = [u for u in self.units if u.team == self.current_player and u.is_alive()]
        if not active_units:
            terminated = True
            return {"obs": self._get_obs(), "action_mask": self._get_action_mask()}, -1.0, terminated, False, {}

        if self.active_unit_idx >= len(active_units):
            self.current_player = 1 - self.current_player
            self.active_unit_idx = 0
            self.turn_count += 1
            if self.turn_count >= self.max_turns:
                return {"obs": self._get_obs(), "action_mask": self._get_action_mask()}, -0.5, True, False, {}

        unit = active_units[self.active_unit_idx]

        # Movimiento
        if action <= 4:
            dx, dy = dirs[action]
            new_pos = (unit.position[0] + dx, unit.position[1] + dy)
            if self._valid_move(new_pos) and not self._position_occupied(new_pos):
                unit.move(new_pos)
        # Ataque
        else:
            delta = (attacks_melee + attacks_archer)[action - 5]
            target = (unit.position[0] + delta[0], unit.position[1] + delta[1])
            for enemy in self.units:
                if enemy.team != self.current_player and enemy.is_alive() and enemy.position == target:
                    damage = unit.get_attack_damage(enemy)
                    enemy.health -= damage
                    reward += 0.2
                    if isinstance(unit, Soldier) and enemy.unit_type == "Archer":
                        reward += 0.2
                    if enemy.health <= 0:
                        reward += 1.0
                    break
            else:
                reward -= 0.1

        if unit.position == self.capture_point:
            self.capture_progress[self.current_player] += 1
            if self.capture_progress[self.current_player] >= self.capture_turns_required:
                return {"obs": self._get_obs(), "action_mask": self._get_action_mask()}, 1.5, True, False, {}
        else:
            self.capture_progress[self.current_player] = 0

        if not any(u.team != self.current_player and u.is_alive() for u in self.units):
            return {"obs": self._get_obs(), "action_mask": self._get_action_mask()}, 1.5, True, False, {}

        self.active_unit_idx += 1
        obs = self._get_obs()
        return {"obs": obs, "action_mask": self._get_action_mask()}, reward, False, False, {}

    def _get_obs(self):
        board = np.zeros((10, *self.board_size), dtype=np.float32)
        for x, y in self.blocked_positions:
            board[0, x, y] = 1.0
        for unit in self.units:
            if unit.is_alive():
                x, y = unit.position
                ch = unit.health / 100
                ch_id = 1 if isinstance(unit, Soldier) else 2
                channel = 1 + (unit.team * 3) + (ch_id - 1)
                board[channel, x, y] = ch
        cx, cy = self.capture_point
        board[7, cx, cy] = 1.0

        for unit in self.units:
            if unit.team == self.current_player and unit.is_alive():
                x, y = unit.position
                board[8, x, y] = 1.0
                board[9, x, y] = 1 if isinstance(unit, Soldier) else 2

        return board

    def _get_action_mask(self):
        mask = np.zeros(9, dtype=np.int8)
        dirs = [(0, 0), (0, -1), (0, 1), (-1, 0), (1, 0)]
        attacks_melee = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        attacks_archer = [(dx, dy) for dx in range(-3, 4) for dy in range(-3, 4)
                          if abs(dx) + abs(dy) in [2, 3] and (dx == 0 or dy == 0)]

        active_units = [u for u in self.units if u.team == self.current_player and u.is_alive()]
        if self.active_unit_idx >= len(active_units):
            return mask

        unit = active_units[self.active_unit_idx]
        for a, (dx, dy) in enumerate(dirs):
            nx, ny = unit.position[0] + dx, unit.position[1] + dy
            if 0 <= nx < self.board_size[0] and 0 <= ny < self.board_size[1] and (nx, ny) not in self.blocked_positions:
                if not self._position_occupied((nx, ny)) or (dx, dy) == (0, 0):
                    mask[a] = 1

        attack_deltas = attacks_archer if unit.unit_type == "Archer" else attacks_melee
        for i, (dx, dy) in enumerate(attack_deltas):
            if i >= 4: break
            tx, ty = unit.position[0] + dx, unit.position[1] + dy
            if 0 <= tx < self.board_size[0] and 0 <= ty < self.board_size[1]:
                for e in self.units:
                    if e.team != self.current_player and e.is_alive() and e.position == (tx, ty):
                        mask[5 + i] = 1
                        break

        return mask

    def _valid_move(self, pos):
        x, y = pos
        return 0 <= x < self.board_size[0] and 0 <= y < self.board_size[1] and pos not in self.blocked_positions

    def _position_occupied(self, pos):
        return any(u.is_alive() and u.position == pos for u in self.units)

    def _has_path(self, start, goal):
        visited = {start}
        queue = deque([start])
        while queue:
            x, y = queue.popleft()
            if (x, y) == goal:
                return True
            for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
                nx, ny = x + dx, y + dy
                if (0 <= nx < self.board_size[0] and 0 <= ny < self.board_size[1]
                        and (nx, ny) not in self.blocked_positions and (nx, ny) not in visited):
                    visited.add((nx, ny))
                    queue.append((nx, ny))
        return False