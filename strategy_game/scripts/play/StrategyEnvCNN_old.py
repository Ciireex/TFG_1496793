
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
import networkx as nx
from gym_strategy.core.Unit import Soldier, Archer, Knight

class StrategyEnv_CNN(gym.Env):
    def __init__(self, use_obstacles=True):
        super().__init__()
        self.board_size = (7, 5)
        self.capture_point = (3, 2)
        self.max_turns = 60
        self.capture_turns_required = 3
        self.unit_types = [Soldier, Soldier, Archer, Knight] * 2
        self.num_units = 8
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(0, 1, shape=(21, 7, 5), dtype=np.float32)
        self.use_obstacles = use_obstacles
        self.reset()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.turn_count = 0
        self.current_player = 0
        self.unit_index_per_team = {0: 0, 1: 0}
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

        if self.use_obstacles:
            self.obstacles = self._generate_obstacles([u.position for u in self.units])
        else:
            self.obstacles = np.zeros(self.board_size, dtype=np.int8)

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
        was_on_point = unit.position == self.capture_point

        if self.phase == "move":
            dx, dy = dirs[action]
            new_pos = (unit.position[0] + dx, unit.position[1] + dy)
            if self._valid_move(new_pos):
                unit.move(new_pos)
                if was_on_point and new_pos != self.capture_point:
                    reward -= 0.3
            else:
                reward -= 0.3
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
                    if unit.unit_type == "Knight":
                        px, py = enemy.position[0] + dx, enemy.position[1] + dy
                        if self._valid_move((px, py)):
                            enemy.move((px, py))
                            enemy.health -= unit.get_attack_damage(enemy)
                        else:
                            enemy.health -= unit.get_attack_damage(enemy) + 10
                    else:
                        enemy.health -= unit.get_attack_damage(enemy)

                    reward += 0.2

                    if enemy.position == self.capture_point and unit.position != self.capture_point:
                        self.attacked_unit_on_point_by_team[enemy.team] = True
                    attacked = True
                    break
            if attacked:
                break

        if not attacked:
            reward -= 0.2

        self._advance_phase()

        if self.turn_count >= self.max_turns:
            reward -= 1.0
            terminated = True

        if hasattr(self, '_terminated') and self._terminated:
            terminated = True
            reward += getattr(self, '_last_reward', 0.0)
            del self._terminated
            del self._last_reward

        return self._get_obs(), reward, terminated, False, {}

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
        team = self.current_player
        on_point_unit = next((u for u in self.units if u.team == team and u.is_alive() and u.position == self.capture_point), None)
        if on_point_unit:
            enemies_alive = any(u.team != team and u.is_alive() for u in self.units)
            if self.attacked_unit_on_point_by_team[team] and enemies_alive:
                self.capture_progress[team] = 0
            else:
                self.capture_progress[team] += 1
            self.attacked_unit_on_point_by_team[team] = False
        else:
            self.capture_progress[team] = 0
            self.attacked_unit_on_point_by_team[team] = False

        if self.capture_progress[team] >= self.capture_turns_required:
            self._terminated = True
            self._last_reward = 3.0
            return

        self.current_player = 1 - self.current_player
        self.unit_index_per_team[self.current_player] = 0
        self.turn_count += 1
        self.phase = "move"

        if self.turn_count >= self.max_turns:
            self._terminated = True
            self._last_reward = -1.0

    def _valid_coord(self, pos):
        x, y = pos
        return 0 <= x < self.board_size[0] and 0 <= y < self.board_size[1]

    def _valid_move(self, pos):
        return self._valid_coord(pos) and self.obstacles[pos] == 0 and not any(u.position == pos and u.is_alive() for u in self.units)

    def _get_obs(self):
        obs = np.zeros((21, self.board_size[0], self.board_size[1]), dtype=np.float32)

        for x in range(self.board_size[0]):
            for y in range(self.board_size[1]):
                if self.obstacles[x, y]:
                    obs[0, x, y] = 1.0

        for unit in self.units:
            if unit.is_alive():
                x, y = unit.position
                team_idx = 1 if unit.team == self.current_player else 4
                type_base = 2 if unit.team == self.current_player else 7
                hp_idx = 3 if unit.team == self.current_player else 6
                obs[team_idx, x, y] = 1.0
                if unit.unit_type == "Soldier":
                    obs[type_base + 0, x, y] = 1.0
                elif unit.unit_type == "Knight":
                    obs[type_base + 1, x, y] = 1.0
                elif unit.unit_type == "Archer":
                    obs[type_base + 2, x, y] = 1.0
                obs[hp_idx, x, y] = unit.health / 100.0

        my_units = [u for u in self.units if u.team == self.current_player and u.is_alive()]
        idx = self.unit_index_per_team[self.current_player]
        if idx < len(my_units):
            ux, uy = my_units[idx].position
            obs[13, ux, uy] = 1.0
            if my_units[idx].unit_type == "Soldier":
                obs[16, ux, uy] = 1.0
            elif my_units[idx].unit_type == "Knight":
                obs[17, ux, uy] = 1.0
            elif my_units[idx].unit_type == "Archer":
                obs[18, ux, uy] = 1.0
            dxs = [(0, -1), (0, 1), (-1, 0), (1, 0)]
            for dx, dy in dxs:
                for dist in range(1, 4 if my_units[idx].unit_type == "Archer" else 2):
                    tx, ty = ux + dx * dist, uy + dy * dist
                    if not self._valid_coord((tx, ty)):
                        break
                    for enemy in self.units:
                        if enemy.team != self.current_player and enemy.is_alive() and enemy.position == (tx, ty):
                            obs[14, ux, uy] = 1.0

        obs[8, :, :] = 1.0 if self.phase == "attack" else 0.0
        obs[9, :, :] = float(self.current_player)
        cx, cy = self.capture_point
        obs[10, cx, cy] = 1.0
        obs[11, :, :] = self.turn_count / self.max_turns
        for x in range(self.board_size[0]):
            for y in range(self.board_size[1]):
                if self._valid_move((x, y)):
                    obs[12, x, y] = 1.0
        if any(u.team != self.current_player and u.position == self.capture_point and u.is_alive() for u in self.units):
            obs[15, cx, cy] = 1.0
        if self.capture_progress[1 - self.current_player] > 0 and                 any(u.team != self.current_player and u.is_alive() and u.position == self.capture_point for u in self.units):
            obs[19, :, :] = 1.0
        mask = self.valid_action_mask()
        for i in range(self.action_space.n):
            obs[20, :, :] += (1.0 if mask[i] else 0.0) * (i + 1) / self.action_space.n
        return obs

    def _generate_obstacles(self, units_positions, obstacle_count=4):
        max_attempts = 100
        half_width = self.board_size[0] // 2
        left_half = [(x, y) for x in range(1, half_width - 1) for y in range(1, self.board_size[1] - 1)]
        right_half = [(x, y) for x in range(half_width + 1, self.board_size[0] - 1) for y in range(1, self.board_size[1] - 1)]

        for _ in range(max_attempts):
            obstacles = np.zeros(self.board_size, dtype=np.int8)
            occupied = set(units_positions + [self.capture_point])
            valid_left = [pos for pos in left_half if pos not in occupied]
            valid_right = [pos for pos in right_half if pos not in occupied]
            if len(valid_left) < obstacle_count // 2 or len(valid_right) < obstacle_count // 2:
                continue
            sampled = random.sample(valid_left, obstacle_count // 2) + random.sample(valid_right, obstacle_count // 2)
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
        raise Exception("No se pudo generar un mapa equilibrado con obstáculos.")

    def valid_action_mask(self):
        mask = np.zeros(self.action_space.n, dtype=bool)
        dirs = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]
        team_units = [u for u in self.units if u.team == self.current_player and u.is_alive()]
        if not team_units:
            mask[0] = True
            return mask
        idx = self.unit_index_per_team[self.current_player]
        if idx >= len(team_units):
            mask[0] = True
            return mask
        unit = team_units[idx]
        if self.phase == "move":
            for i, (dx, dy) in enumerate(dirs):
                new_pos = (unit.position[0] + dx, unit.position[1] + dy)
                if i == 0:
                    mask[i] = True
                elif self._valid_coord(new_pos) and self._valid_move(new_pos):
                    mask[i] = True
        elif self.phase == "attack":
            attack_range = 3 if unit.unit_type == "Archer" else 1
            for i, (dx, dy) in enumerate(dirs):
                if i == 0:
                    continue
                for dist in range(1, attack_range + 1):
                    tx, ty = unit.position[0] + dx * dist, unit.position[1] + dy * dist
                    if not self._valid_coord((tx, ty)):
                        break
                    for enemy in self.units:
                        if enemy.team != self.current_player and enemy.is_alive() and enemy.position == (tx, ty):
                            mask[i] = True
                            break
                    if mask[i]:
                        break
            if not any(mask[1:]):
                mask[0] = True
        return mask
