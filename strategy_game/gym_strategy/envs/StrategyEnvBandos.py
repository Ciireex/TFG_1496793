import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
from collections import deque
from gym_strategy.core.Unit import Soldier, Archer

class StrategyEnvBandos(gym.Env):
    def __init__(self, blue_team=None, red_team=None):
        super().__init__()
        self.board_size = (9, 6)  # ancho > alto, estilo Advance Wars
        self.max_turns = 60
        self.capture_turns_required = 3

        self.blue_team = blue_team if blue_team else [Soldier, Soldier, Archer]
        self.red_team = red_team if red_team else [Soldier, Archer, Soldier]
        self.unit_types = self.blue_team + self.red_team
        self.num_units = len(self.unit_types)

        self.action_space = spaces.MultiDiscrete([9] * (self.num_units // 2))
        self.observation_space = spaces.Dict({
            "obs": spaces.Box(0.0, 1.0, shape=(8, *self.board_size), dtype=np.float32),
            "action_mask": spaces.MultiBinary((self.num_units // 2, 9))
        })

        self.reset()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.turn_count = 0
        self.current_player = 0
        self.capture_progress = [0, 0]
        self.previous_positions = {}
        self.last_distances = {}

        width, height = self.board_size
        while True:
            num_blocks = random.randint(6, 10)
            all_cells = [(x, y) for x in range(width) for y in range(height)]
            self.blocked_positions = set(random.sample(all_cells, num_blocks))
            free = [c for c in all_cells if c not in self.blocked_positions]

            blue_side = [(0, y) for y in range(height)]
            red_side = [(width - 1, y) for y in range(height)]
            center_options = [(x, height // 2) for x in range(3, 6)]

            blue_spawns = [c for c in blue_side if c in free]
            red_spawns = [c for c in red_side if c in free]
            centers = [c for c in center_options if c in free]

            if len(blue_spawns) >= 3 and len(red_spawns) >= 3 and centers:
                self.spawn_blue = random.sample(blue_spawns, 3)
                self.spawn_red = random.sample(red_spawns, 3)
                self.capture_point = random.choice(centers)

                all_spawns = self.spawn_blue + self.spawn_red
                if all(self._has_path(s, self.capture_point) for s in all_spawns):
                    break

        self.units = []
        for i, unit_cls in enumerate(self.unit_types):
            team = 0 if i < 3 else 1
            spawn = self.spawn_blue[i] if team == 0 else self.spawn_red[i - 3]
            self.units.append(unit_cls(position=spawn, team=team))

        obs = self._get_obs()
        return {"obs": obs, "action_mask": self._get_action_mask()}, {}

    def step(self, actions):
        reward = -0.05
        terminated = False

        active_units = [u for u in self.units if u.team == self.current_player and u.is_alive()]
        enemy_units = [u for u in self.units if u.team != self.current_player and u.is_alive()]

        dirs = [(0, 0), (0, -1), (0, 1), (-1, 0), (1, 0)]
        attacks = [(0, -1), (0, 1), (-1, 0), (1, 0)]

        for i, (unit, action) in enumerate(zip(active_units, actions)):
            key = (self.current_player, i)
            last_dist = self.last_distances.get(key)

            dist_before = min([self._manhattan_distance(unit.position, e.position) for e in enemy_units], default=10)

            if action <= 4:
                dx, dy = dirs[action]
                new_pos = (unit.position[0] + dx, unit.position[1] + dy)
                if self._valid_move(new_pos) and not self._position_occupied(new_pos):
                    unit.move(new_pos)
                    self.previous_positions[key] = new_pos
            else:
                dx, dy = attacks[action - 5]
                target = (unit.position[0] + dx, unit.position[1] + dy)
                for enemy in enemy_units:
                    if enemy.is_alive() and enemy.position == target:
                        damage = unit.get_attack_damage(enemy)
                        enemy.health -= damage
                        reward += 0.4
                        if isinstance(unit, Soldier) and enemy.unit_type == "Archer":
                            reward += 0.2
                        break
                else:
                    reward -= 0.2  # penaliza fallar ataque

            dist_after = min([self._manhattan_distance(unit.position, e.position) for e in enemy_units], default=10)
            self.last_distances[key] = dist_after

            if dist_after < dist_before:
                reward += 0.02
            elif dist_after > dist_before:
                reward -= 0.05

            if unit.position == self.capture_point:
                if self.capture_progress[self.current_player] > 0:
                    reward += 0.1  # solo si está manteniendo la captura
                self.capture_progress[self.current_player] += 1
                if self.capture_progress[self.current_player] >= self.capture_turns_required:
                    reward = 1.5
                    terminated = True
            else:
                self.capture_progress[self.current_player] = 0

        if not any(u.team != self.current_player and u.is_alive() for u in self.units):
            reward = 1.5
            terminated = True

        self.turn_count += 1
        if self.turn_count >= self.max_turns:
            reward -= 0.5  # penalización por no resolver la partida
            terminated = True

        if not terminated:
            self.current_player = 1 - self.current_player

        obs = self._get_obs()
        return {"obs": obs, "action_mask": self._get_action_mask()}, reward, terminated, False, {"episode": {"r": reward, "l": self.turn_count}} if terminated else {}

    def _get_obs(self):
        board = np.zeros((8, *self.board_size), dtype=np.float32)
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
        return board

    def _get_action_mask(self):
        mask = np.zeros((3, 9), dtype=np.int8)
        dirs = [(0, 0), (0, -1), (0, 1), (-1, 0), (1, 0)]
        attacks = [(0, -1), (0, 1), (-1, 0), (1, 0)]

        my_units = [u for u in self.units if u.team == self.current_player and u.is_alive()]
        enemies = [u for u in self.units if u.team != self.current_player and u.is_alive()]

        for i, unit in enumerate(my_units):
            for a, (dx, dy) in enumerate(dirs):
                nx, ny = unit.position[0] + dx, unit.position[1] + dy
                if 0 <= nx < self.board_size[0] and 0 <= ny < self.board_size[1] and (nx, ny) not in self.blocked_positions:
                    if not self._position_occupied((nx, ny)) or (dx, dy) == (0, 0):
                        mask[i, a] = 1
            for a, (dx, dy) in enumerate(attacks):
                tx, ty = unit.position[0] + dx, unit.position[1] + dy
                if 0 <= tx < self.board_size[0] and 0 <= ty < self.board_size[1]:
                    for e in enemies:
                        if e.position == (tx, ty):
                            mask[i, 5 + a] = 1
                            break
        return mask

    def _valid_move(self, pos):
        x, y = pos
        return 0 <= x < self.board_size[0] and 0 <= y < self.board_size[1] and pos not in self.blocked_positions

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
                if (0 <= nx < self.board_size[0] and 0 <= ny < self.board_size[1]
                        and (nx, ny) not in self.blocked_positions and (nx, ny) not in visited):
                    visited.add((nx, ny))
                    queue.append((nx, ny))
        return False
