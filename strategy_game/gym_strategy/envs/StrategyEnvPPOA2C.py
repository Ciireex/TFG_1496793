import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
import networkx as nx
from gym_strategy.core.Unit import Soldier, Archer, Knight

class StrategyEnvPPOA2C(gym.Env):
    def __init__(self):
        super().__init__()
        self.board_size = (7, 5)
        self.capture_point = (3, 2)
        self.max_turns = 60
        self.capture_turns_required = 3

        self.unit_types = [Soldier, Soldier, Archer, Knight] * 2
        self.num_units = 8

        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(0, 1, shape=(14, 7, 5), dtype=np.float32)

        self.reset()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.turn_count = 0
        self.current_player = 0
        self.active_unit_index = 0
        self.phase = "move"
        self.capture_progress = [0, 0]
        self.attacked_unit_on_point_by_team = [False, False]
        self.units = []

        blue_spawns = [(0, 1), (0, 2), (0, 3), (0, 4)]
        red_spawns = [(6, 1), (6, 2), (6, 3), (6, 4)]

        for i in range(8):
            team = 0 if i < 4 else 1
            pos = blue_spawns[i] if team == 0 else red_spawns[i - 4]
            unit = self.unit_types[i](position=pos, team=team)
            self.units.append(unit)

        self.obstacles = self._generate_obstacles([u.position for u in self.units])
        return self._get_obs(), {}

    def step(self, action):
        reward = 0.0
        terminated = False

        dirs = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]
        team_units = [u for u in self.units if u.team == self.current_player and u.is_alive()]
        if not team_units:
            self._advance_turn()
            return self._get_obs(), reward, False, False, {}

        if self.active_unit_index >= len(team_units):
            self._advance_turn()
            team_units = [u for u in self.units if u.team == self.current_player and u.is_alive()]

        unit = team_units[self.active_unit_index]
        was_on_point = unit.position == self.capture_point

        if self.phase == "move":
            dx, dy = dirs[action]
            new_pos = (unit.position[0] + dx, unit.position[1] + dy)

            if self._valid_move(new_pos):
                unit.move(new_pos)
                if was_on_point and new_pos != self.capture_point:
                    reward -= 1.0
            else:
                reward -= 0.5

            self.phase = "attack"
            return self._get_obs(), reward, False, False, {}

        dx, dy = dirs[action]
        attacked = False
        for dist in range(1, 4 if unit.unit_type == "Archer" else 2):
            tx, ty = unit.position[0] + dx * dist, unit.position[1] + dy * dist
            if not self._valid_coord((tx, ty)):
                break
            for enemy in self.units:
                if enemy.team != self.current_player and enemy.is_alive() and enemy.position == (tx, ty):
                    # Incentivo por atacar según debilidad
                    bonus = 0.0
                    if unit.unit_type == "Soldier" and enemy.unit_type == "Archer":
                        bonus = 0.2
                    elif unit.unit_type == "Archer" and enemy.unit_type == "Knight":
                        bonus = 0.2
                    elif unit.unit_type == "Knight" and enemy.unit_type == "Soldier":
                        bonus = 0.2

                    if unit.unit_type == "Knight":
                        px, py = enemy.position[0] + dx, enemy.position[1] + dy
                        if self._valid_move((px, py)):
                            enemy.move((px, py))
                            enemy.health -= unit.get_attack_damage(enemy)
                        else:
                            enemy.health -= unit.get_attack_damage(enemy) + 10
                    else:
                        enemy.health -= unit.get_attack_damage(enemy)

                    reward += 0.5 + bonus

                    if enemy.position == self.capture_point:
                        if unit.position != self.capture_point:
                            self.attacked_unit_on_point_by_team[enemy.team] = True

                    if not enemy.is_alive():
                        reward += 2.0
                    attacked = True
                    break
            if attacked:
                break

        if not attacked:
            reward -= 0.5

        self._advance_phase()

        if self.phase == "move" and self.active_unit_index == 0:
            team = self.current_player
            on_point_unit = next((u for u in self.units if u.team == team and u.is_alive() and u.position == self.capture_point), None)

            if on_point_unit:
                if self.attacked_unit_on_point_by_team[team]:
                    self.capture_progress[team] = 0
                    self.attacked_unit_on_point_by_team[team] = False
                else:
                    self.capture_progress[team] += 1
            else:
                self.capture_progress[team] = 0
                self.attacked_unit_on_point_by_team[team] = False

            if self.capture_progress[team] >= self.capture_turns_required:
                reward += 3.0
                terminated = True

        teams_alive = set(u.team for u in self.units if u.is_alive())
        if len(teams_alive) == 1:
            if self.current_player in teams_alive:
                reward += 3.0
            terminated = True

        if self.turn_count >= self.max_turns:
            reward -= 1.5
            terminated = True

        return self._get_obs(), reward, terminated, False, {}

    def _advance_phase(self):
        if self.phase == "move":
            self.phase = "attack"
        else:
            self.phase = "move"
            self.active_unit_index += 1
            if self.active_unit_index >= len([u for u in self.units if u.team == self.current_player and u.is_alive()]):
                self._advance_turn()

    def _advance_turn(self):
        self.current_player = 1 - self.current_player
        self.active_unit_index = 0
        self.turn_count += 1
        self.phase = "move"

    def _manhattan(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _valid_coord(self, pos):
        x, y = pos
        return 0 <= x < self.board_size[0] and 0 <= y < self.board_size[1]

    def _valid_move(self, pos):
        return self._valid_coord(pos) and self.obstacles[pos] == 0 and not any(u.position == pos and u.is_alive() for u in self.units)

    def _get_obs(self):
        obs = np.zeros((14, self.board_size[0], self.board_size[1]), dtype=np.float32)

        for x in range(self.board_size[0]):
            for y in range(self.board_size[1]):
                if self.obstacles[x, y]:
                    obs[0, x, y] = 1.0

        for unit in self.units:
            if unit.is_alive():
                x, y = unit.position
                idx = 1 if unit.team == self.current_player else 4
                type_idx = 2 if unit.team == self.current_player else 5
                hp_idx = 3 if unit.team == self.current_player else 6
                obs[idx, x, y] = 1.0
                obs[type_idx, x, y] = (
                    1.0 if unit.unit_type == "Archer" else 0.75 if unit.unit_type == "Knight" else 0.5
                )
                obs[hp_idx, x, y] = unit.health / 100.0

        my_units = [u for u in self.units if u.team == self.current_player and u.is_alive()]
        if self.active_unit_index < len(my_units):
            ux, uy = my_units[self.active_unit_index].position
            obs[7, ux, uy] = 1.0

            dxs = [(0, -1), (0, 1), (-1, 0), (1, 0)]
            for dx, dy in dxs:
                for dist in range(1, 4 if my_units[self.active_unit_index].unit_type == "Archer" else 2):
                    tx, ty = ux + dx * dist, uy + dy * dist
                    if not self._valid_coord((tx, ty)):
                        break
                    for enemy in self.units:
                        if enemy.team != self.current_player and enemy.is_alive() and enemy.position == (tx, ty):
                            obs[13, ux, uy] = 1.0

        obs[8, :, :] = 1.0 if self.phase == "attack" else 0.0
        obs[9, :, :] = float(self.current_player)
        cx, cy = self.capture_point
        obs[10, cx, cy] = 1.0
        obs[11, :, :] = self.turn_count / self.max_turns

        for x in range(self.board_size[0]):
            for y in range(self.board_size[1]):
                if self._valid_move((x, y)):
                    obs[12, x, y] = 1.0

        return obs

    def _generate_obstacles(self, units_positions, obstacle_count=3):
        attempts = 100
        for _ in range(attempts):
            obstacles = np.zeros(self.board_size, dtype=np.int8)
            free = [(x, y) for x in range(self.board_size[0]) for y in range(self.board_size[1])
                    if (x, y) not in units_positions and (x, y) != self.capture_point]
            if len(free) < obstacle_count:
                continue
            sampled = random.sample(free, obstacle_count)
            for x, y in sampled:
                obstacles[x, y] = 1

            G = nx.grid_2d_graph(*self.board_size)
            for x, y in sampled:
                G.remove_node((x, y))

            try:
                if all(nx.has_path(G, pos, self.capture_point) for pos in units_positions):
                    return obstacles
            except:
                continue
        raise Exception("No se pudo generar obstáculos válidos.")
