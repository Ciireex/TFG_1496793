import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
from collections import deque
from gym_strategy.core.Unit import Soldier, Archer

class StrategyEnvA2CSplit2v2(gym.Env):
    def __init__(self):
        super().__init__()
        self.board_size = (6, 4)
        self.max_turns = 60
        self.capture_turns_required = 3

        self.unit_types = [Soldier, Archer] * 2  # 2 vs 2
        self.action_space = spaces.Discrete(5)  # 0: quieto, 1-4: direcciones
        self.observation_space = spaces.Box(0, 255, shape=(12, 4, 6), dtype=np.uint8)

        self.phase = "move"
        self.reset()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.turn_count = 0
        self.current_player = 0
        self.active_unit_index = 0
        self.capture_progress = [0, 0]
        self.phase = "move"

        width, height = self.board_size
        while True:
            all_cells = [(x, y) for x in range(width) for y in range(height)]
            self.blocked_positions = set(random.sample(all_cells, random.randint(2, 4)))
            free = [c for c in all_cells if c not in self.blocked_positions]

            blue_side = [(0, y) for y in range(height)]
            red_side = [(width - 1, y) for y in range(height)]
            center = (width // 2, height // 2)

            blue_spawns = [c for c in blue_side if c in free]
            red_spawns = [c for c in red_side if c in free]

            if len(blue_spawns) >= 2 and len(red_spawns) >= 2 and center in free:
                self.spawn_blue = random.sample(blue_spawns, 2)
                self.spawn_red = random.sample(red_spawns, 2)
                self.capture_point = center
                if all(self._has_path(s, self.capture_point) for s in self.spawn_blue + self.spawn_red):
                    break

        self.units = []
        for i, unit_cls in enumerate(self.unit_types):
            team = 0 if i < 2 else 1
            spawn = self.spawn_blue[i] if team == 0 else self.spawn_red[i - 2]
            self.units.append(unit_cls(position=spawn, team=team))

        return self._get_obs(), {}

    def step(self, action):
        reward = -0.01
        terminated = False
        info = {}

        team_units = [u for u in self.units if u.team == self.current_player and u.is_alive()]
        if len(team_units) == 0:
            self._advance_turn()
            return self._get_obs(), reward, False, False, info

        if self.active_unit_index >= len(team_units):
            self._advance_turn()
            team_units = [u for u in self.units if u.team == self.current_player and u.is_alive()]

        unit = team_units[self.active_unit_index]
        move_dirs = [(0, 0), (0, -1), (0, 1), (-1, 0), (1, 0)]

        if self.current_player == 1:  # Heurística roja
            if self.phase == "move":
                dx, dy = self._bfs_step(unit.position, self.capture_point)
                new_pos = (unit.position[0] + dx, unit.position[1] + dy)
                if self._valid_move(new_pos) and not self._position_occupied(new_pos):
                    unit.move(new_pos)
            else:
                for dx, dy in move_dirs[1:]:
                    tx, ty = unit.position[0] + dx, unit.position[1] + dy
                    for enemy in self.units:
                        if enemy.team != self.current_player and enemy.is_alive() and enemy.position == (tx, ty):
                            enemy.health -= unit.get_attack_damage(enemy)
                            break
            self._advance_phase()
            return self._get_obs(), reward, False, False, info

        else:  # IA azul
            if self.phase == "move":
                dx, dy = move_dirs[action]
                new_pos = (unit.position[0] + dx, unit.position[1] + dy)
                if self._valid_move(new_pos) and not self._position_occupied(new_pos):
                    unit.move(new_pos)
                    reward += 0.02
                else:
                    reward -= 0.1
                self.phase = "attack"
                return self._get_obs(), reward, False, False, {}
            else:
                atk_dirs = move_dirs[1:]
                dx, dy = atk_dirs[action - 1] if action > 0 else (0, 0)
                target = (unit.position[0] + dx, unit.position[1] + dy)
                for enemy in self.units:
                    if enemy.team != self.current_player and enemy.is_alive() and enemy.position == target:
                        enemy.health -= unit.get_attack_damage(enemy)
                        reward += 0.5
                        if enemy.health <= 0:
                            reward += 1.0
                        break
                else:
                    reward -= 0.05
                self._advance_phase()

        if unit.position == self.capture_point:
            self.capture_progress[self.current_player] += 1
            reward += 0.3
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

        if terminated:
            info = {"episode": {"r": reward, "l": self.turn_count, "winner": self.current_player}}

        return self._get_obs(), reward, terminated, False, info

    def _advance_phase(self):
        self.phase = "move" if self.phase == "attack" else "move"
        if self.phase == "move":
            self.active_unit_index += 1
            if self.active_unit_index >= len([u for u in self.units if u.team == self.current_player and u.is_alive()]):
                self._advance_turn()

    def _advance_turn(self):
        self.current_player = 1 - self.current_player
        self.active_unit_index = 0
        self.turn_count += 1
        self.phase = "move"

    def _bfs_step(self, start, goal):
        queue = deque([(start, [])])
        visited = {start}
        while queue:
            current, path = queue.popleft()
            if current == goal:
                return path[0] if path else (0, 0)
            for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
                nx, ny = current[0] + dx, current[1] + dy
                new = (nx, ny)
                if self._valid_move(new) and not self._position_occupied(new) and new not in visited:
                    visited.add(new)
                    queue.append((new, path + [(dx, dy)]))
        return (0, 0)

    def _get_obs(self):
        board = np.zeros((12, *self.board_size), dtype=np.float32)
        for x, y in self.blocked_positions:
            board[0, x, y] = 1.0
        for unit in self.units:
            if unit.is_alive():
                x, y = unit.position
                ch = unit.health / 100
                unit_idx = {"Soldier": 1, "Archer": 2}[unit.unit_type]
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
