import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
import networkx as nx
from gym_strategy.core.Unit import Soldier, Archer, Knight

UNIT_CLASSES = {
    "Soldier": Soldier,
    "Archer": Archer,
    "Knight": Knight
}

class StrategyEnv_V3(gym.Env):
    def __init__(self, team_blue_units=None, team_red_units=None, use_obstacles=True):
        super().__init__()
        self.board_size = (7, 5)
        self.capture_point = (3, 2)
        self.max_turns = 60
        self.capture_turns_required = 3
        self.use_obstacles = use_obstacles

        self.team_blue_units = team_blue_units if team_blue_units else ["Soldier", "Soldier", "Archer", "Knight"]
        self.team_red_units = team_red_units if team_red_units else ["Soldier", "Soldier", "Archer", "Knight"]

        self.unit_types = self.team_blue_units + self.team_red_units
        self.num_units = len(self.unit_types)

        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(0, 1, shape=(14, 7, 5), dtype=np.float32)
        self.reset()

    def team_has_units(self, team):
        return any(u.team == team and u.is_alive() for u in self.units)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.turn_count = 0
        self.current_player = 0
        self.unit_index_per_team = {0: 0, 1: 0}
        self.phase = "move"
        self.capture_progress = [0, 0]
        self.attacked_unit_on_point_by_team = [False, False]
        self.units = []

        blue_positions = self._generate_spawn_positions(side="left", count=len(self.team_blue_units))
        red_positions = self._generate_spawn_positions(side="right", count=len(self.team_red_units)) if self.team_red_units else []

        for i, unit_type in enumerate(self.team_blue_units):
            pos = blue_positions[i]
            unit = UNIT_CLASSES[unit_type](position=pos, team=0)
            self.units.append(unit)

        for i, unit_type in enumerate(self.team_red_units):
            pos = red_positions[i]
            unit = UNIT_CLASSES[unit_type](position=pos, team=1)
            self.units.append(unit)

        occupied = [u.position for u in self.units]
        if self.use_obstacles:
            self.obstacles = self._generate_obstacles(occupied)
        else:
            self.obstacles = np.zeros(self.board_size, dtype=np.int8)

        return self._get_obs(), {}

    def _generate_spawn_positions(self, side, count):
        columns = [0, 1] if side == "left" else [5, 6]
        all_positions = [(x, y) for x in columns for y in range(self.board_size[1])]
        return random.sample(all_positions, k=count)

    def _valid_coord(self, pos):
        x, y = pos
        return 0 <= x < self.board_size[0] and 0 <= y < self.board_size[1]

    def _valid_move(self, pos):
        return self._valid_coord(pos) and self.obstacles[pos] == 0 and not any(u.position == pos and u.is_alive() for u in self.units)

    def _advance_turn(self):
        self.current_player = 1 - self.current_player
        self.unit_index_per_team[self.current_player] = 0
        self.turn_count += 1
        self.phase = "move"

    def _advance_phase(self):
        if self.phase == "move":
            self.phase = "attack"
        else:
            self.phase = "move"
            self.unit_index_per_team[self.current_player] += 1
            team_units = [u for u in self.units if u.team == self.current_player and u.is_alive()]
            if self.unit_index_per_team[self.current_player] >= len(team_units):
                self._advance_turn()

    def _generate_obstacles(self, occupied_positions, obstacle_count=4):
        max_attempts = 100
        half_width = self.board_size[0] // 2

        left_half = [(x, y) for x in range(1, half_width)
                    for y in range(1, self.board_size[1] - 1)]
        right_half = [(x, y) for x in range(half_width, self.board_size[0] - 1)
                    for y in range(1, self.board_size[1] - 1)]

        for _ in range(max_attempts):
            obstacles = np.zeros(self.board_size, dtype=np.int8)
            occupied = set(occupied_positions + [self.capture_point])

            valid_left = [pos for pos in left_half if pos not in occupied]
            valid_right = [pos for pos in right_half if pos not in occupied]

            if len(valid_left) < obstacle_count // 2 or len(valid_right) < obstacle_count // 2:
                continue

            sampled_left = random.sample(valid_left, obstacle_count // 2)
            sampled_right = random.sample(valid_right, obstacle_count // 2)
            sampled = sampled_left + sampled_right

            for x, y in sampled:
                obstacles[x, y] = 1

            # Comprobar conectividad
            G = nx.grid_2d_graph(*self.board_size)
            for x, y in sampled:
                if G.has_node((x, y)):
                    G.remove_node((x, y))

            try:
                if all(nx.has_path(G, pos, self.capture_point) for pos in occupied_positions):
                    return obstacles
            except:
                continue

        raise Exception("No se pudo generar un mapa estilo Advance Wars equilibrado.")

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
                obs[type_idx, x, y] = 1.0 if unit.unit_type == "Archer" else 0.75 if unit.unit_type == "Knight" else 0.5
                obs[hp_idx, x, y] = unit.health / 100.0

        my_units = [u for u in self.units if u.team == self.current_player and u.is_alive()]
        idx = self.unit_index_per_team[self.current_player]
        if idx < len(my_units):
            ux, uy = my_units[idx].position
            obs[7, ux, uy] = 1.0
            dxs = [(0, -1), (0, 1), (-1, 0), (1, 0)]
            for dx, dy in dxs:
                for dist in range(1, 4 if my_units[idx].unit_type == "Archer" else 2):
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

    def step(self, action):
        reward = 0.0
        terminated = False

        dirs = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]
        team_units = [u for u in self.units if u.team == self.current_player and u.is_alive()]
        if not team_units:
            self._advance_turn()
            return self._get_obs(), reward, False, False, {}

        index = self.unit_index_per_team[self.current_player]
        if index >= len(team_units):
            self._advance_turn()
            team_units = [u for u in self.units if u.team == self.current_player and u.is_alive()]
            index = self.unit_index_per_team[self.current_player]
            if not team_units:
                return self._get_obs(), reward, False, False, {}

        unit = team_units[index]
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

                    if enemy.position == self.capture_point and unit.position != self.capture_point:
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

        if self.phase == "move" and self.unit_index_per_team[self.current_player] == 0:
            team = self.current_player
            on_point = next((u for u in self.units if u.team == team and u.is_alive() and u.position == self.capture_point), None)

            if on_point:
                if self.attacked_unit_on_point_by_team[team]:
                    self.capture_progress[team] = 0
                    self.attacked_unit_on_point_by_team[team] = False
                else:
                    self.capture_progress[team] += 1
            else:
                self.capture_progress[team] = 0
                self.attacked_unit_on_point_by_team[team] = False

            if self.capture_progress[team] == self.capture_turns_required:
                if not hasattr(self, "reward_given"):
                    self.reward_given = [False, False]
                if not self.reward_given[team]:
                    reward += 10.0
                    self.reward_given[team] = True

            if self.capture_progress[team] >= self.capture_turns_required:
                reward += 3.0
                terminated = True

        if self.turn_count >= self.max_turns:
            reward -= 1.5
            terminated = True

        return self._get_obs(), reward, terminated, False, {}

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
