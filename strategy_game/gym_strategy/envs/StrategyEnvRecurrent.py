import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
from collections import deque
from gym_strategy.core.Unit import Soldier, Archer

class StrategyEnvRecurrent(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self, blue_team=None, red_team=None):
        super().__init__()
        self.board_size = (9, 6)
        self.max_turns = 60
        self.capture_turns_required = 3

        self.blue_team = blue_team if blue_team else [Soldier, Soldier, Archer]
        self.red_team = red_team if red_team else [Archer, Soldier, Soldier]
        self.unit_types = self.blue_team + self.red_team
        self.num_units = len(self.unit_types)

        obs_shape = (11 * self.board_size[0] * self.board_size[1],)
        self.observation_space = spaces.Box(0.0, 1.0, shape=obs_shape, dtype=np.float32)
        self.action_space = spaces.Discrete(5)
        self.reset()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.turn_count = 0
        self.current_player = 0
        self.active_unit_index = 0
        self.phase = "move"
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
        return obs.flatten(), {}

    def step(self, action):
        reward = 0.0
        terminated = False

        team_units = [u for u in self.units if u.team == self.current_player and u.is_alive()]
        if self.active_unit_index >= len(team_units):
            self._advance_turn()
            team_units = [u for u in self.units if u.team == self.current_player and u.is_alive()]

        unit = team_units[self.active_unit_index]
        dirs = [(0, 0), (0, -1), (0, 1), (-1, 0), (1, 0)]
        attacks_melee = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        attacks_archer = [(dx, dy) for dx in range(-3, 4) for dy in range(-3, 4)
                          if abs(dx) + abs(dy) in [2, 3] and (dx == 0 or dy == 0)]
        enemies = [u for u in self.units if u.team != self.current_player and u.is_alive()]

        if self.phase == "move":
            dx, dy = dirs[action]
            new_pos = (unit.position[0] + dx, unit.position[1] + dy)
            if self._valid_move(new_pos) and not self._position_occupied(new_pos):
                unit.move(new_pos)
                reward += 0.01
            else:
                reward -= 0.05
            self.phase = "attack"
        else:
            hit_successful = False
            attack_deltas = attacks_archer if unit.unit_type == "Archer" else attacks_melee
            if action < len(attack_deltas):
                dx, dy = attack_deltas[action]
                target = (unit.position[0] + dx, unit.position[1] + dy)
                for enemy in enemies:
                    if enemy.position == target and enemy.is_alive():
                        damage = unit.get_attack_damage(enemy)
                        enemy.health -= damage
                        reward += 0.2
                        hit_successful = True
                        if isinstance(unit, Soldier) and enemy.unit_type == "Archer":
                            reward += 0.2
                        if enemy.health <= 0:
                            reward += 0.8
                        break
                if not hit_successful:
                    reward -= 0.05
            else:
                reward -= 0.05

            if unit.position == self.capture_point:
                self.capture_progress[self.current_player] += 1
                if self.capture_progress[self.current_player] >= self.capture_turns_required:
                    reward += 2.0
                    terminated = True
            else:
                self.capture_progress[self.current_player] = 0

            if not any(u.team != self.current_player and u.is_alive() for u in self.units):
                reward += 2.0
                terminated = True

            if self.turn_count >= self.max_turns:
                reward -= 1.0
                terminated = True

            self.active_unit_index += 1
            if self.active_unit_index >= len(team_units):
                self._advance_turn()
            self.phase = "move"

        obs = self._get_obs()
        winner = self.current_player if reward >= 2.0 else -1
        return obs.flatten(), reward, terminated, False, {
            "episode": {"r": reward, "l": self.turn_count, "winner": winner}
        } if terminated else {}

    def _advance_turn(self):
        self.current_player = 1 - self.current_player
        self.active_unit_index = 0
        self.turn_count += 1
        self.phase = "move"

    def _get_obs(self):
        board = np.zeros((11, *self.board_size), dtype=np.float32)
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

        my_units = [u for u in self.units if u.team == self.current_player and u.is_alive()]
        if self.active_unit_index < len(my_units):
            unit = my_units[self.active_unit_index]
            x, y = unit.position
            board[8, x, y] = 1.0
            board[9, x, y] = 1 if isinstance(unit, Soldier) else 2
        board[10] = self.current_player
        return board

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