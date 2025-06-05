import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
from gym_strategy.core.Unit import Soldier, Archer, Knight

class StrategyEnv_V4(gym.Env):
    def __init__(self, use_obstacles=True, only_blue=False, enemy_controller=None):
        super().__init__()
        self.board_size = (7, 5)
        self.capture_point = (3, 2)
        self.max_turns = 60
        self.capture_turns_required = 3
        self.only_blue = only_blue
        self.enemy_controller = enemy_controller
        self.unit_types = [Soldier, Soldier, Archer, Knight] if only_blue else [Soldier, Soldier, Archer, Knight] * 2
        self.num_units = len(self.unit_types)
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(0, 1, shape=(14, 7, 5), dtype=np.float32)
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
        self.turns_on_point_by_unit = {}
        self.prev_distance_to_capture = {}

        blue_spawns = [(0, 1), (0, 2), (0, 3), (0, 4)]
        red_spawns = [(6, 1), (6, 2), (6, 3), (6, 4)]

        for i in range(self.num_units):
            team = 0 if i < 4 else 1
            if self.only_blue and team == 1:
                continue
            pos = blue_spawns[i] if team == 0 else red_spawns[i - 4]
            unit = self.unit_types[i](position=pos, team=team)
            self.units.append(unit)

        if self.use_obstacles:
            self.obstacles = self._generate_obstacles([u.position for u in self.units])
        else:
            self.obstacles = np.zeros(self.board_size, dtype=np.int8)

        for unit in self.units:
            if unit.team == 0:
                self.prev_distance_to_capture[id(unit)] = self._manhattan(unit.position, self.capture_point)

        return self._get_obs(), {}

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
        prev_pos = unit.position

        if self.enemy_controller is not None and self.current_player == 1:
            action = self._get_enemy_action()

        dx, dy = dirs[action]
        new_pos = (unit.position[0] + dx, unit.position[1] + dy)

        if self.phase == "move":
            if self._valid_move(new_pos):
                unit.move(new_pos)
                if was_on_point and new_pos != self.capture_point:
                    unit_id = id(unit)
                    lost = self.turns_on_point_by_unit.pop(unit_id, 0)
                    if lost > 0:
                        reward -= 0.3 * lost
                    self.capture_progress[self.current_player] = 0
            else:
                reward -= 0.5

        elif self.phase == "attack":
            enemy_hit = None
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                target_pos = (unit.position[0] + dx, unit.position[1] + dy)
                for enemy in self.units:
                    if enemy.is_alive() and enemy.team != self.current_player and enemy.position == target_pos:
                        enemy_hit = enemy
                        break
                if enemy_hit:
                    damage = unit.attack(enemy_hit)
                    if damage and damage > 0:
                        reward += 0.3  # recompensa base por atacar
                        if enemy_hit.position == self.capture_point:
                            reward += 0.5  # bonus por atacar unidad sobre el punto

        self._advance_phase()

        if self.phase == "move" and self.unit_index_per_team[self.current_player] == 0:
            team = self.current_player
            on_point_unit = next((u for u in self.units if u.team == team and u.is_alive() and u.position == self.capture_point), None)
            if on_point_unit:
                unit_id = id(on_point_unit)
                self.turns_on_point_by_unit.setdefault(unit_id, 0)
                self.turns_on_point_by_unit[unit_id] += 1
                self.capture_progress[team] += 1
                reward += 0.3
                if prev_pos == self.capture_point:
                    reward += 0.05  # bonus por no moverse del punto
            else:
                for uid in list(self.turns_on_point_by_unit.keys()):
                    lost = self.turns_on_point_by_unit.pop(uid, 0)
                    if lost > 0:
                        reward -= 0.3 * lost
                self.capture_progress[team] = 0

            if self.capture_progress[team] >= self.capture_turns_required:
                reward += 3.0
                terminated = True
                self.turns_on_point_by_unit.clear()

        if not self.only_blue:
            teams_alive = set(u.team for u in self.units if u.is_alive())
            if len(teams_alive) == 1:
                if self.current_player in teams_alive:
                    reward += 3.0
                terminated = True

        if self.turn_count >= self.max_turns:
            reward -= 1.5
            terminated = True

        return self._get_obs(), reward, terminated, False, {}

    def _manhattan(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _advance_phase(self):
        if self.phase == "move":
            self.phase = "attack" if not self.only_blue else "move"
        else:
            self.phase = "move"
            self.unit_index_per_team[self.current_player] += 1
            team_units = [u for u in self.units if u.team == self.current_player and u.is_alive()]
            if self.unit_index_per_team[self.current_player] >= len(team_units):
                self._advance_turn()

        if self.only_blue and self.phase == "move":
            self.unit_index_per_team[self.current_player] += 1
            team_units = [u for u in self.units if u.team == self.current_player and u.is_alive()]
            if self.unit_index_per_team[self.current_player] >= len(team_units):
                self._advance_turn()

    def _advance_turn(self):
        self.current_player = 1 - self.current_player if not self.only_blue else 0
        self.unit_index_per_team[self.current_player] = 0
        self.turn_count += 1
        self.phase = "move"

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
                obs[type_idx, x, y] = 1.0 if unit.unit_type == "Archer" else 0.75 if unit.unit_type == "Knight" else 0.5
                obs[hp_idx, x, y] = unit.health / 100.0
        my_units = [u for u in self.units if u.team == self.current_player and u.is_alive()]
        idx = self.unit_index_per_team[self.current_player]
        if idx < len(my_units):
            ux, uy = my_units[idx].position
            obs[7, ux, uy] = 1.0
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

    def _generate_obstacles(self, units_positions, obstacle_count=4):
        attempts = 100
        for _ in range(attempts):
            obstacles = np.zeros(self.board_size, dtype=np.int8)
            candidates = [(x, y) for x in range(1, self.board_size[0]-1) for y in range(1, self.board_size[1]-1)]
            valid = [pos for pos in candidates if pos not in units_positions and pos != self.capture_point]
            if len(valid) < obstacle_count:
                continue
            sampled = random.sample(valid, obstacle_count)
            for x, y in sampled:
                obstacles[x, y] = 1
            return obstacles
        return np.zeros(self.board_size, dtype=np.int8)

    def _get_enemy_action(self):
        team_units = [u for u in self.units if u.team == 1 and u.is_alive()]
        index = self.unit_index_per_team[1]
        if index >= len(team_units):
            return 0
        unit = team_units[index]
        best_dir = 0
        min_dist = float('inf')
        dirs = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]
        for i, (dx, dy) in enumerate(dirs):
            new_pos = (unit.position[0] + dx, unit.position[1] + dy)
            if self._valid_move(new_pos):
                dist = abs(new_pos[0] - self.capture_point[0]) + abs(new_pos[1] - self.capture_point[1])
                if dist < min_dist:
                    best_dir = i
                    min_dist = dist
        return best_dir
