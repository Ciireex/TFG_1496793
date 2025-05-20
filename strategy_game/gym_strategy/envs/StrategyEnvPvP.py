import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
from gym_strategy.core.Unit import Soldier, Archer, Knight
from gym_strategy.core.Board import Board
import pygame

class StrategyEnvPvP(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self, render_mode=None):
        super().__init__()
        self.board_size = (7, 5)
        self.capture_point = (3, 2)
        self.capture_progress = [0, 0]
        self.capture_unit_attacked = [False, False]
        self.max_capture = 3

        self.units = []
        self.board = Board(size=self.board_size)
        self.render_mode = render_mode
        self.renderer = None
        self.current_player = 0
        self.active_unit_index = 0
        self.phase = "move"
        self.done = False

        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(low=0, high=1, shape=(16, *self.board_size), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_player = 0
        self.active_unit_index = 0
        self.phase = "move"
        self.done = False
        self.capture_progress = [0, 0]
        self.capture_unit_attacked = [False, False]
        self.units = []
        self.board = Board(size=self.board_size)

        for team in [0, 1]:
            columns = [0, 1] if team == 0 else [5, 6]
            positions = random.sample([(x, y) for x in columns for y in range(self.board_size[1])], 4)
            team_units = [
                Soldier(positions[0], team),
                Soldier(positions[1], team),
                Archer(positions[2], team),
                Knight(positions[3], team),
            ]
            for unit in team_units:
                self.board.add_unit(unit)
                self.units.append(unit)

        return self.get_obs(), {}

    def get_obs(self):
        obs = np.zeros((16, *self.board_size), dtype=np.float32)
        for unit in self.units:
            if not unit.is_alive():
                continue
            x, y = unit.position
            c_base = 0 if unit.team == 0 else 1
            obs[c_base, x, y] = 1
            if unit.unit_type == "Soldier":
                obs[2, x, y] = 1
            elif unit.unit_type == "Archer":
                obs[3, x, y] = 1
            elif unit.unit_type == "Knight":
                obs[4, x, y] = 1
            obs[5, x, y] = unit.health / 100

        unit = self.get_active_unit()
        if unit and unit.is_alive():
            x, y = unit.position
            obs[6, x, y] = 1
            obs[7, :, :] = unit.team

        obs[14, :, :] = 1.0 if self.phase == "attack" else 0.0

        cx, cy = self.capture_point
        obs[15, cx, cy] = 1.0

        return obs

    def get_active_unit(self):
        team_units = [u for u in self.units if u.team == self.current_player and u.is_alive()]
        if self.active_unit_index < len(team_units):
            return team_units[self.active_unit_index]
        return None

    def step(self, action):
        if self.done:
            return self.get_obs(), 0.0, True, False, {}

        unit = self.get_active_unit()
        if unit is None:
            self.end_turn()
            return self.get_obs(), 0.0, self.done, False, {}

        reward = 0.0

        if self.phase == "move":
            move_vec = self.get_direction(action)
            if move_vec and unit.movement >= 1:
                new_pos = (unit.position[0] + move_vec[0], unit.position[1] + move_vec[1])
                if self.board.is_valid_move(new_pos):
                    self.board.move_unit(unit, new_pos)
            self.phase = "attack"
            return self.get_obs(), 0.0, False, False, {}

        elif self.phase == "attack":
            atk_vec = self.get_direction(action)
            if atk_vec:
                target = self.find_target(unit, atk_vec)
                if target:
                    initial_hp = target.health
                    unit.attack(target)
                    damage = max(0, initial_hp - target.health)
                    reward += 0.1 * (damage / 100)
                    if not target.is_alive():
                        reward += 0.2
                        self.board.grid[target.position] = 0
                    if target.position == self.capture_point:
                        self.capture_unit_attacked[target.team] = True

            dead_units = [u for u in self.units if not u.is_alive()]
            for dead in dead_units:
                self.board.grid[dead.position] = 0
            self.units = [u for u in self.units if u.is_alive()]

            allies = [u for u in self.units if u.team == self.current_player]
            enemies = [u for u in self.units if u.team != self.current_player]

            if not enemies:
                self.done = True
                return self.get_obs(), 1.0 + reward, True, False, {}
            if not allies:
                self.done = True
                return self.get_obs(), -1.0, True, False, {}

            self.active_unit_index += 1
            if self.active_unit_index >= len([u for u in self.units if u.team == self.current_player]):
                self.end_turn()
            else:
                self.phase = "move"

        return self.get_obs(), reward, self.done, False, {}

    def end_turn(self):
        team = self.current_player
        units_on_point = [u for u in self.units if u.team == team and u.position == self.capture_point and u.is_alive()]

        if units_on_point and not self.capture_unit_attacked[team]:
            self.capture_progress[team] += 1
        else:
            self.capture_progress[team] = 0

        self.capture_unit_attacked = [False, False]

        if self.capture_progress[team] >= self.max_capture:
            self.done = True
            return

        self.current_player = 1 - team
        self.active_unit_index = 0
        self.phase = "move"

    def get_direction(self, direction_index):
        if direction_index == 0:
            return None
        return [(0, -1), (1, 0), (0, 1), (-1, 0)][direction_index - 1]

    def find_target(self, attacker, direction):
        dx, dy = direction
        x, y = attacker.position
        for dist in range(1, 4 if attacker.unit_type == "Archer" else 2):
            tx = x + dx * dist
            ty = y + dy * dist
            if not (0 <= tx < self.board_size[0] and 0 <= ty < self.board_size[1]):
                break
            for unit in self.units:
                if unit.position == (tx, ty) and unit.team != attacker.team and unit.is_alive():
                    return unit
            if attacker.unit_type != "Archer":
                break
        return None

    def render(self):
        if self.render_mode == "human":
            if self.renderer is None:
                from gym_strategy.core.Renderer import Renderer
                self.renderer = Renderer(width=700, height=500, board_size=self.board_size)
            self.renderer.draw_board(
                self.units,
                active_unit=self.get_active_unit(),
                capture_point=self.capture_point,
                capture_score=self.capture_progress,
                max_capture=self.max_capture
            )

    def close(self):
        if self.renderer:
            pygame.quit()
