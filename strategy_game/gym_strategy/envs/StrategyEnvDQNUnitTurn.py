import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
from collections import deque
from gym_strategy.core.Unit import Soldier, Archer, Knight

class StrategyEnvDQNUnitTurn(gym.Env):
    def __init__(self, team_controlled=0, opponent_model=None):
        super().__init__()
        self.board_size = (9, 6)
        self.max_turns = 60
        self.capture_turns_required = 3

        self.team_controlled = team_controlled
        self.opponent_model = opponent_model

        self.unit_types = [Soldier, Soldier, Archer, Knight] * 2
        self.num_units = len(self.unit_types)

        self.action_space = spaces.Discrete(25)
        self.observation_space = spaces.Box(0, 255, shape=(12, 6, 9), dtype=np.uint8)

        self.reset()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.turn_count = 0
        self.current_player = 0
        self.active_unit_index = 0
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

            if len(blue_spawns) >= 4 and len(red_spawns) >= 4 and centers:
                self.spawn_blue = random.sample(blue_spawns, 4)
                self.spawn_red = random.sample(red_spawns, 4)
                self.capture_point = random.choice(centers)

                if all(self._has_path(s, self.capture_point) for s in self.spawn_blue + self.spawn_red):
                    break

        self.units = []
        for i, unit_cls in enumerate(self.unit_types):
            team = 0 if i < 4 else 1
            spawn = self.spawn_blue[i] if team == 0 else self.spawn_red[i - 4]
            self.units.append(unit_cls(position=spawn, team=team))

        return self._get_obs(), {}

    def step(self, action):
        reward = -0.01
        terminated = False
        info = {}

        team_units = [u for u in self.units if u.team == self.current_player and u.is_alive()]
        if self.active_unit_index >= len(team_units):
            self._advance_turn()
            team_units = [u for u in self.units if u.team == self.current_player and u.is_alive()]

        if len(team_units) == 0:
            self._advance_turn()
            return self._get_obs(), reward, False, False, info

        unit = team_units[self.active_unit_index]

        if self.current_player != self.team_controlled:
            obs = self._get_obs()
            action = self.opponent_model.predict(obs, deterministic=True)[0] if self.opponent_model else 0

        move_id = action // 5
        atk_id = action % 5

        move_dirs = [(0, 0), (0, -1), (0, 1), (-1, 0), (1, 0)]
        atk_dirs = [(0, 0), (0, -1), (0, 1), (-1, 0), (1, 0)]

        dx, dy = move_dirs[move_id]
        new_pos = (unit.position[0] + dx, unit.position[1] + dy)
        if self._valid_move(new_pos) and not self._position_occupied(new_pos):
            unit.move(new_pos)
            if move_id != 0:
                reward += 0.02
        else:
            reward -= 0.1

        dx, dy = atk_dirs[atk_id]
        if (dx, dy) != (0, 0):
            target = (unit.position[0] + dx, unit.position[1] + dy)
            for enemy in self.units:
                if enemy.team != self.current_player and enemy.is_alive() and enemy.position == target:
                    damage = unit.get_attack_damage(enemy)
                    if isinstance(unit, Soldier) and enemy.unit_type == "Archer":
                        damage *= 1.5
                    elif isinstance(unit, Archer) and enemy.unit_type == "Knight":
                        damage *= 1.5
                    elif isinstance(unit, Knight) and enemy.unit_type == "Soldier":
                        damage *= 1.5
                    enemy.health -= damage
                    reward += 0.5
                    if enemy.health <= 0:
                        reward += 1.5
                    break
            else:
                reward -= 0.1

        dist_to_capture = abs(unit.position[0] - self.capture_point[0]) + abs(unit.position[1] - self.capture_point[1])
        reward += max(0, 0.1 - 0.01 * dist_to_capture)

        if unit.position == self.capture_point:
            self.capture_progress[self.current_player] += 1
            reward += 0.3
            if self.capture_progress[self.current_player] >= self.capture_turns_required:
                reward += 2.0
                terminated = True
        else:
            self.capture_progress[self.current_player] = 0

        if not any(u.team != self.current_player and u.is_alive() for u in self.units):
            reward += 3.0
            terminated = True

        if self.turn_count >= self.max_turns:
            reward -= 1.0
            terminated = True

        self.active_unit_index += 1
        if self.active_unit_index >= len(team_units):
            self._advance_turn()

        if terminated:
            info = {"episode": {"r": reward, "l": self.turn_count, "winner": self.current_player}}

        return self._get_obs(), reward, terminated, False, info

    def _get_obs(self):
        board = np.zeros((12, *self.board_size), dtype=np.float32)
        for x, y in self.blocked_positions:
            board[0, x, y] = 1.0
        for unit in self.units:
            if unit.is_alive():
                x, y = unit.position
                ch = unit.health / 100
                unit_idx = {"Soldier": 1, "Archer": 2, "Knight": 3}[unit.unit_type]
                ch_base = 1 + (unit.team * 3) + (unit_idx - 1)
                board[ch_base, x, y] = ch
        cx, cy = self.capture_point
        board[10, cx, cy] = 1.0

        my_units = [u for u in self.units if u.team == self.current_player and u.is_alive()]
        if self.active_unit_index < len(my_units):
            unit = my_units[self.active_unit_index]
            x, y = unit.position
            board[11, x, y] = 1.0

        return (board.transpose(0, 2, 1) * 255).astype(np.uint8)

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

    def _advance_turn(self):
        self.current_player = 1 - self.current_player
        self.active_unit_index = 0
        self.turn_count += 1
