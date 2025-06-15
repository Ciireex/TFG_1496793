import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from gym_strategy.core.Unit import Soldier, Archer

class StrategyEnv_CastleControl(gym.Env):
    def __init__(self):
        super().__init__()
        self.board_size = (8, 6)
        self.max_turns = 80
        self.unit_types = [Soldier, Soldier, Archer, Archer] * 2
        self.num_units = 8

        # CASTILLO: posiciones centrales
        self.castle_tiles = [(3, 2), (3, 3), (4, 2), (4, 3)]
        self.castle_score = 0  # entre -5 y +5
        self.castle_limit = 5

        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(0, 1, shape=(25, *self.board_size), dtype=np.float32)
        self.reset()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.turn_count = 0
        self.current_player = 0
        self.unit_index_per_team = {0: 0, 1: 0}
        self.phase = "move"
        self.units = []
        self.castle_score = 0

        blue_positions = [(0, i) for i in range(4)]
        red_positions = [(7, i) for i in range(4)]

        for i, unit_class in enumerate(self.unit_types[:4]):
            self.units.append(unit_class(position=blue_positions[i], team=0))
        for i, unit_class in enumerate(self.unit_types[4:]):
            self.units.append(unit_class(position=red_positions[i], team=1))

        self.obstacles = np.zeros(self.board_size, dtype=np.int8)
        self.terrain = {
            'forest': np.zeros(self.board_size, dtype=np.int8),
            'camp': np.zeros(self.board_size, dtype=np.int8),
            'hill': np.zeros(self.board_size, dtype=np.int8),
        }

        return self._get_obs(), {}

    def step(self, action):
        reward = 0.0
        terminated = False

        dirs = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]
        team_units = [u for u in self.units if u.team == self.current_player and u.is_alive()]
        index = self.unit_index_per_team[self.current_player]

        if index >= len(team_units):
            self._advance_turn()
            return self._get_obs(), reward, False, False, {}

        unit = team_units[index]

        if self.phase == "move":
            dx, dy = dirs[action]
            new_pos = (unit.position[0] + dx, unit.position[1] + dy)
            if not self._valid_move(new_pos):
                reward -= 0.01
            else:
                unit.move(new_pos)
            self.phase = "attack"
            return self._get_obs(), reward, False, False, {}

        dx, dy = dirs[action]
        attacked = False

        # Atacar enemigos o castillo
        min_range = getattr(unit, "min_range", 1)
        for dist in range(min_range, unit.attack_range + 1):
            tx, ty = unit.position[0] + dx * dist, unit.position[1] + dy * dist
            if not self._valid_coord((tx, ty)):
                break

            # Atacar castillo
            if (tx, ty) in self.castle_tiles:
                delta = 1 if self.current_player == 0 else -1
                self.castle_score = np.clip(self.castle_score + delta, -self.castle_limit, self.castle_limit)
                reward += 0.5  # Más recompensa que atacar enemigo
                if self.castle_score == self.castle_limit or self.castle_score == -self.castle_limit:
                    reward += 2.0
                    terminated = True
                attacked = True
                break

            # Atacar enemigo
            for enemy in self.units:
                if enemy.team != self.current_player and enemy.is_alive() and enemy.position == (tx, ty):
                    enemy.health -= unit.get_attack_damage(enemy)
                    reward += 0.1
                    if not enemy.is_alive():
                        reward += 0.3
                    attacked = True
                    break

            if attacked:
                break

        if not attacked:
            reward -= 0.1

        self._advance_phase()

        if self.turn_count >= self.max_turns:
            reward -= 2.0
            terminated = True

        return self._get_obs(), reward, terminated, False, {}

    def _valid_coord(self, pos):
        x, y = pos
        return 0 <= x < self.board_size[0] and 0 <= y < self.board_size[1]

    def _valid_move(self, pos):
        return (
            self._valid_coord(pos)
            and self.obstacles[pos] == 0
            and pos not in self.castle_tiles
            and not any(u.position == pos and u.is_alive() for u in self.units)
        )

    def _advance_phase(self):
        if self.phase == "move":
            self.phase = "attack"
        else:
            self.phase = "move"
            self.unit_index_per_team[self.current_player] += 1
            team_units = [u for u in self.units if u.team == self.current_player and u.is_alive()]
            if self.unit_index_per_team[self.current_player] >= len(team_units):
                self._advance_turn()

    def _advance_turn(self):
        self.current_player = 1 - self.current_player
        self.unit_index_per_team[self.current_player] = 0
        self.turn_count += 1
        self.phase = "move"

    def _get_active_unit(self):
        team_units = [u for u in self.units if u.team == self.current_player and u.is_alive()]
        if self.unit_index_per_team[self.current_player] < len(team_units):
            return team_units[self.unit_index_per_team[self.current_player]]
        return None

    def _get_obs(self):
        obs = np.zeros((25, *self.board_size), dtype=np.float32)

        obs[0] = self.obstacles
        obs[1] = self.terrain['forest']
        obs[2] = self.terrain['camp']
        obs[3] = self.terrain['hill']

        for unit in self.units:
            if not unit.is_alive():
                continue
            x, y = unit.position
            if unit.team == self.current_player:
                obs[4, x, y] = 1.0         # Posición aliado
                obs[5, x, y] = 1.0         # ¿Soldado?
                obs[6, x, y] = unit.health / 100.0
            else:
                obs[7, x, y] = 1.0         # Posición enemigo
                obs[8, x, y] = 1.0         # ¿Soldado?
                obs[9, x, y] = unit.health / 100.0

        active_unit = self._get_active_unit()
        if active_unit:
            x, y = active_unit.position
            obs[10, x, y] = 1.0
            dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
            for d_index, (dx, dy) in enumerate(dirs):
                for dist in range(1, active_unit.attack_range + 1):
                    tx, ty = x + dx * dist, y + dy * dist
                    if not self._valid_coord((tx, ty)):
                        break
                    obs[11 + d_index, tx, ty] = 1.0

        obs[15, :, :] = 1.0 if self.phase == "attack" else 0.0

        # Estado del castillo normalizado en canal 16
        norm_score = (self.castle_score + self.castle_limit) / (2 * self.castle_limit)
        obs[16, :, :] = norm_score

        # Posiciones del castillo en canal 17
        for x, y in self.castle_tiles:
            obs[17, x, y] = 1.0

        return obs