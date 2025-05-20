import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
import networkx as nx
from gym_strategy.core.Unit import Soldier, Archer, Knight

class StrategyEnvAj(gym.Env):
    def __init__(self):
        super().__init__()
        self.board_size = (7, 5)
        self.capture_point = (3, 2)
        self.max_turns = 60
        self.capture_turns_required = 3

        self.unit_types = [Soldier, Soldier, Archer, Knight] * 2
        self.num_units = 8

        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(0, 1, shape=(18, *self.board_size), dtype=np.float32)

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
                    reward -= 0.1
            else:
                reward -= 0.2

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
                    bonus = 0.0

                    if enemy.position == self.capture_point:
                        bonus += 0.5
                        self.attacked_unit_on_point_by_team[enemy.team] = True

                    if enemy.health < unit.health:
                        bonus += 0.2

                    damage = unit.get_attack_damage(enemy)

                    if unit.unit_type == "Knight":
                        px, py = enemy.position[0] + dx, enemy.position[1] + dy
                        if self._valid_move((px, py)):
                            enemy.move((px, py))
                            enemy.health -= damage
                        else:
                            enemy.health -= damage + 10
                    else:
                        enemy.health -= damage

                    reward += 0.2 + bonus

                    if not enemy.is_alive():
                        reward += 0.5

                    attacked = True
                    break
            if attacked:
                break

        if not attacked:
            reward -= 0.1

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
            elif self.capture_progress[1 - team] >= self.capture_turns_required:
                reward -= 2.0
                terminated = True

        teams_alive = set(u.team for u in self.units if u.is_alive())
        if len(teams_alive) == 1:
            if self.current_player in teams_alive:
                reward += 2.0
            else:
                reward -= 2.0
            terminated = True

        if self.turn_count >= self.max_turns:
            reward -= 1.0
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

    def _valid_coord(self, pos):
        x, y = pos
        return 0 <= x < self.board_size[0] and 0 <= y < self.board_size[1]

    def _valid_move(self, pos):
        return self._valid_coord(pos) and self.obstacles[pos] == 0 and not any(u.position == pos and u.is_alive() for u in self.units)

    def _get_obs(self):
        obs = np.zeros((18, *self.board_size), dtype=np.float32)

        obs[0] = self.obstacles

        for unit in self.units:
            if unit.is_alive():
                x, y = unit.position
                if unit.team == self.current_player:
                    obs[1, x, y] = 1.0
                    obs[2, x, y] = 1.0 if unit.unit_type == "Archer" else 0.75 if unit.unit_type == "Knight" else 0.5
                    obs[3, x, y] = unit.health / 100.0
                else:
                    obs[4, x, y] = 1.0
                    obs[5, x, y] = 1.0 if unit.unit_type == "Archer" else 0.75 if unit.unit_type == "Knight" else 0.5
                    obs[6, x, y] = unit.health / 100.0

        my_units = [u for u in self.units if u.team == self.current_player and u.is_alive()]
        if self.active_unit_index < len(my_units):
            ux, uy = my_units[self.active_unit_index].position
            unit_type = my_units[self.active_unit_index].unit_type
            obs[7, ux, uy] = 1.0

            for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
                max_range = 3 if unit_type == "Archer" else 1
                for dist in range(1, max_range + 1):
                    tx, ty = ux + dx * dist, uy + dy * dist
                    if self._valid_coord((tx, ty)):
                        obs[13, tx, ty] = 1.0

        obs[8, :, :] = 1.0 if self.phase == "attack" else 0.0
        obs[9, :, :] = float(self.current_player)
        cx, cy = self.capture_point
        obs[10, cx, cy] = 1.0
        obs[11, :, :] = self.turn_count / self.max_turns

        if self.active_unit_index < len(my_units):
            ux, uy = my_units[self.active_unit_index].position
            for dx, dy in [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]:
                tx, ty = ux + dx, uy + dy
                if self._valid_move((tx, ty)):
                    obs[12, tx, ty] = 1.0

        obs[14, :, :] = self.capture_progress[self.current_player] / self.capture_turns_required
        obs[15, :, :] = self.capture_progress[1 - self.current_player] / self.capture_turns_required

        for unit in self.units:
            if unit.is_alive() and unit.team != self.current_player and unit.position == self.capture_point:
                x, y = unit.position
                obs[16, x, y] = 1.0

        # Canal 17: enemigos en rango de ataque válido (según tipo y distancia)
        if self.active_unit_index < len(my_units):
            ux, uy = my_units[self.active_unit_index].position
            unit_type = my_units[self.active_unit_index].unit_type
            valid_ranges = [1, 2, 3] if unit_type == "Archer" else [1]

            for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
                for dist in valid_ranges:
                    tx, ty = ux + dx * dist, uy + dy * dist
                    if not self._valid_coord((tx, ty)):
                        break
                    for enemy in self.units:
                        if enemy.team != self.current_player and enemy.is_alive() and enemy.position == (tx, ty):
                            obs[17, tx, ty] = 1.0
                            break

        return obs

    def _generate_obstacles(self, units_positions, obstacle_count=3):
        for _ in range(100):
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
