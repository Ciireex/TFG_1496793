import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gym_strategy.core.Unit import Soldier, Archer

class StrategyEnv_Fase3(gym.Env):
    def __init__(self, obstacle_count=6):
        super().__init__()
        self.board_size = (8, 6)
        self.max_turns = 80
        self.unit_types = [Soldier, Soldier, Archer, Archer] * 2
        self.num_units = 8
        self.obstacle_count = obstacle_count
        self.castle_area = []
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(0, 1, shape=(24, *self.board_size), dtype=np.float32)
        self.reset()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.turn_count = 0
        self.current_player = 0
        self.unit_index_per_team = {0: 0, 1: 0}
        self.phase = "move"
        self.units = []

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

        if self.obstacle_count > 0:
            self._place_random_obstacles()
        return self._get_obs(), {}

    def _place_random_obstacles(self):
        positions = [(x, y) for x in range(1, self.board_size[0] - 1)
                            for y in range(self.board_size[1])]
        np.random.shuffle(positions)
        count = 0
        for pos in positions:
            if all(unit.position != pos for unit in self.units):
                self.obstacles[pos] = 1
                count += 1
            if count >= self.obstacle_count:
                break

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
            if not self._valid_coord(new_pos) or self.obstacles[new_pos] == 1:
                reward -= 0.2
            elif any(u.position == new_pos and u.is_alive() for u in self.units):
                reward -= 0.05
            else:
                if new_pos != unit.position:
                    reward += 0.05  # Recompensa por moverse útilmente
                unit.move(new_pos)
            self.phase = "attack"
            return self._get_obs(), reward, False, False, {}

        dx, dy = dirs[action]
        attacked = False
        enemy_in_range = False

        min_range = getattr(unit, "min_range", 1)
        for dist in range(min_range, unit.attack_range + 1):
            tx, ty = unit.position[0] + dx * dist, unit.position[1] + dy * dist
            if not self._valid_coord((tx, ty)):
                break
            for enemy in self.units:
                if enemy.team != self.current_player and enemy.is_alive() and enemy.position == (tx, ty):
                    enemy_in_range = True
                    enemy.health -= unit.get_attack_damage(enemy)
                    attacked = True
                    reward += 0.3
                    if not enemy.is_alive():
                        reward += 0.7
                        if self._team_won(self.current_player):
                            reward += 1.0
                            terminated = True
                    break
            if attacked:
                break

        if enemy_in_range and not attacked:
            reward -= 0.2

        self._advance_phase()

        if self.turn_count >= self.max_turns:
            reward -= 1.0
            terminated = True

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
        self.current_player = 1 - self.current_player
        self.unit_index_per_team[self.current_player] = 0
        self.turn_count += 1
        self.phase = "move"

    def _valid_coord(self, pos):
        x, y = pos
        return 0 <= x < self.board_size[0] and 0 <= y < self.board_size[1]

    def _valid_move(self, pos):
        return (
            self._valid_coord(pos)
            and self.obstacles[pos] == 0
            and not any(u.position == pos and u.is_alive() for u in self.units)
        )

    def _team_won(self, team_id):
        return all(not u.is_alive() for u in self.units if u.team != team_id)

    def _get_active_unit(self):
        team_units = [u for u in self.units if u.team == self.current_player and u.is_alive()]
        if self.unit_index_per_team[self.current_player] < len(team_units):
            return team_units[self.unit_index_per_team[self.current_player]]
        return None

    def _get_obs(self):
        obs = np.zeros((24, *self.board_size), dtype=np.float32)
        obs[0] = self.obstacles
        obs[1] = self.terrain['forest']
        obs[2] = self.terrain['camp']
        obs[3] = self.terrain['hill']

        for unit in self.units:
            if not unit.is_alive():
                continue
            x, y = unit.position
            if unit.team == self.current_player:
                obs[4, x, y] = 1.0
                obs[5, x, y] = 1.0
                obs[6, x, y] = unit.health / 100.0
            else:
                obs[7, x, y] = 1.0
                obs[8, x, y] = 1.0
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
                    for enemy in self.units:
                        if enemy.team != self.current_player and enemy.is_alive() and enemy.position == (tx, ty):
                            obs[11, tx, ty] = 1.0
                            obs[12 + d_index, tx, ty] = 1.0
                            break

            unit_type = 1 if isinstance(active_unit, Soldier) else 2
            obs[17, x, y] = unit_type / 2.0

            for dx, dy in dirs:
                tx, ty = x + dx, y + dy
                if self._valid_move((tx, ty)):
                    obs[18, tx, ty] = 1.0

        obs[16, :, :] = 1.0 if self.phase == "attack" else 0.0
        return obs
