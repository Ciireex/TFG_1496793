import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
from collections import deque
from gym_strategy.core.Unit import Soldier

class StrategyEnvSoldiersOnly(gym.Env):
    def __init__(self):
        super().__init__()
        self.board_size = (9, 6)
        self.max_turns = 60
        self.capture_turns_required = 3

        self.unit_types = [Soldier] * 6  # 3 por equipo
        self.num_units = len(self.unit_types)

        # 5 movimientos (quieto + 4 direcciones) x 5 ataques (ninguno + 4 direcciones)
        self.action_space = spaces.Discrete(25)
        self.observation_space = spaces.Box(0.0, 1.0, shape=(11, *self.board_size), dtype=np.float32)

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

        return self._get_obs(), {}

    def step(self, action):
        reward = 0.0
        terminated = False

        team_units = [u for u in self.units if u.team == self.current_player and u.is_alive()]
        if self.active_unit_index >= len(team_units):
            self._advance_turn()
            team_units = [u for u in self.units if u.team == self.current_player and u.is_alive()]

        unit = team_units[self.active_unit_index]
        move_dirs = [(0, 0), (0, -1), (0, 1), (-1, 0), (1, 0)]  # quieto, arriba, abajo, izq, der
        atk_dirs = [(0, 0), (0, -1), (0, 1), (-1, 0), (1, 0)]  # sin atacar, ↑, ↓, ←, →

        move_id = action // 5
        atk_id = action % 5

        # Movimiento
        dx, dy = move_dirs[move_id]
        new_pos = (unit.position[0] + dx, unit.position[1] + dy)
        if self._valid_move(new_pos) and not self._position_occupied(new_pos):
            unit.move(new_pos)
            reward += 0.01
        else:
            reward -= 0.1

        # Ataque
        dx, dy = atk_dirs[atk_id]
        if (dx, dy) != (0, 0):
            target = (unit.position[0] + dx, unit.position[1] + dy)
            for enemy in self.units:
                if enemy.team != self.current_player and enemy.is_alive() and enemy.position == target:
                    damage = unit.get_attack_damage(enemy)
                    enemy.health -= damage
                    reward += 0.2
                    if enemy.unit_type == "Archer":
                        reward += 0.2
                    if enemy.health <= 0:
                        reward += 0.8
                    break

        # Captura
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

        return self._get_obs(), reward, terminated, False, {
            "episode": {"r": reward, "l": self.turn_count, "winner": self.current_player if reward >= 2.0 else -1}
        } if terminated else {}

    def _get_obs(self):
        board = np.zeros((11, *self.board_size), dtype=np.float32)
        for x, y in self.blocked_positions:
            board[0, x, y] = 1.0
        for unit in self.units:
            if unit.is_alive():
                x, y = unit.position
                board[1 + unit.team * 3, x, y] = unit.health / 100
        cx, cy = self.capture_point
        board[7, cx, cy] = 1.0

        my_units = [u for u in self.units if u.team == self.current_player and u.is_alive()]
        if self.active_unit_index < len(my_units):
            unit = my_units[self.active_unit_index]
            x, y = unit.position
            board[8, x, y] = 1.0  # unidad activa
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

    def _advance_turn(self):
        self.current_player = 1 - self.current_player
        self.active_unit_index = 0
        self.turn_count += 1
