import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.policies import ActorCriticCnnPolicy
import os

# --- Custom CNN Feature Extractor ---
# Make sure this class is in a file accessible to your main script,
# for example, in gym_strategy/utils/CustomCNN.py as you mentioned.
# For this example, I'll include it directly.
class CustomCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        # We assume the observation space is (n_input_channels, height, width)
        n_input_channels = observation_space.shape[0]

        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # Reduces spatial dimensions by half (e.g., 10x6 -> 5x3)
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )

        # Compute the flattened size automatically for the linear layer
        with torch.no_grad():
            # Create a dummy input with the correct shape to pass through CNN
            # observation_space.shape[1:] gives (height, width)
            sample_input = torch.as_tensor(observation_space.sample()[None]).float()
            n_flatten = self.cnn(sample_input).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
            nn.Dropout(p=0.2)  # Soft regularization
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))

# --- Your Game Environment (StrategyEnv_Gemini) ---
# Assuming these Unit classes and StrategyEnv_Gemini are defined as provided in your prompt.
# I'll include them directly here for completeness.
class Unit:
    def __init__(self, position, team, unit_type, health=100, attack_damage=10):
        self.position = position
        self.team = team
        self.unit_type = unit_type
        self.health = health
        self.attack_damage = attack_damage

    def is_alive(self):
        return self.health > 0

    def move(self, new_position):
        self.position = new_position

    def get_attack_damage(self, target_unit=None):
        return self.attack_damage

class Soldier(Unit):
    def __init__(self, position, team):
        super().__init__(position, team, "Soldier", health=100, attack_damage=15)

class Archer(Unit):
    def __init__(self, position, team):
        super().__init__(position, team, "Archer", health=70, attack_damage=10)

class Knight(Unit):
    def __init__(self, position, team):
        super().__init__(position, team, "Knight", health=120, attack_damage=20)


class StrategyEnv_Gemini(gym.Env):
    metadata = {'render_modes': ['human'], 'render_fps': 4}

    def __init__(self, use_obstacles=True, obstacle_count=10):
        super().__init__()
        self.board_size = (10, 6)
        self.castle_area = [(4, 2), (4, 3), (5, 2), (5, 3)]
        self.max_turns = 60
        self.castle_control = 0
        self.unit_types = [Soldier, Soldier, Archer, Knight] * 2
        self.num_units = 8
        self.action_space = spaces.Discrete(5) # 0: stay, 1: up, 2: down, 3: left, 4: right

        self.num_obs_layers = 22
        self.observation_space = spaces.Box(0, 1, shape=(self.num_obs_layers, *self.board_size), dtype=np.float32)

        self.use_obstacles = use_obstacles
        self.obstacle_count = obstacle_count
        self.reset()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.turn_count = 0
        self.current_player = 0 # 0 for Blue, 1 for Red
        self.unit_index_per_team = {0: 0, 1: 0} # Index of the unit currently acting for each team
        self.phase = "move" # "move" or "attack"
        self.castle_control = 0 # Positive for Blue control, negative for Red control
        self.units = []

        blue_positions = [(0, y) for y in range(1, 4)] + [(1, y) for y in range(1, 4)]
        red_positions = [(self.board_size[0]-1, y) for y in range(1, 4)] + [(self.board_size[0]-2, y) for y in range(1, 4)]
        random.shuffle(blue_positions)
        random.shuffle(red_positions)

        for i, unit_class in enumerate(self.unit_types[:4]):
            self.units.append(unit_class(position=blue_positions[i], team=0))
        for i, unit_class in enumerate(self.unit_types[4:]):
            self.units.append(unit_class(position=red_positions[i], team=1))

        unit_positions = [u.position for u in self.units]
        self.obstacles = self._generate_obstacles(unit_positions, self.obstacle_count) if self.use_obstacles else np.zeros(self.board_size, dtype=np.int8)

        self._update_unit_actions_cache()

        return self._get_obs(), {}

    def step(self, action):
        reward = -0.005
        terminated = False
        info = {}

        dirs = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]
        
        team_units = [u for u in self.units if u.team == self.current_player and u.is_alive()]
        index = self.unit_index_per_team[self.current_player]

        if index >= len(team_units):
            self._advance_turn()
            self._update_unit_actions_cache()
            return self._get_obs(), reward, False, False, info

        unit = team_units[index]

        if self.phase == "move":
            dx, dy = dirs[action]
            new_pos = (unit.position[0] + dx, unit.position[1] + dy)
            if self._valid_move(new_pos):
                unit.move(new_pos)
            else:
                reward -= 0.1
            self.phase = "attack"
            self._update_unit_actions_cache()
            return self._get_obs(), reward, False, False, info

        dx, dy = dirs[action]
        attacked = False
        attack_range = 3 if unit.unit_type == "Archer" else 1

        target_found = False
        for dist in range(1, attack_range + 1):
            tx, ty = unit.position[0] + dx * dist, unit.position[1] + dy * dist
            if not self._valid_coord((tx, ty)):
                break

            if (tx, ty) in self.castle_area:
                prev_control = self.castle_control
                if self.current_player == 0:
                    self.castle_control = min(5, self.castle_control + 1)
                else:
                    self.castle_control = max(-5, self.castle_control - 1)

                if self.castle_control != prev_control:
                    reward += 0.3
                attacked = True
                target_found = True
                break
            
            for enemy in self.units:
                if enemy.team != self.current_player and enemy.is_alive() and enemy.position == (tx, ty):
                    enemy.health -= unit.get_attack_damage(enemy)
                    
                    if enemy.health <= 0:
                        reward += 0.2
                    else:
                        reward += 0.05
                    attacked = True
                    target_found = True
                    break

            if target_found:
                break

        if not attacked and action != 0:
            reward -= 0.02

        for unit_check in self.units:
            if unit_check.team == self.current_player and not unit_check.is_alive() and unit_check not in team_units:
                reward -= 0.1

        self._advance_phase()
        self._update_unit_actions_cache()

        if self.turn_count >= self.max_turns:
            reward -= 1.0
            terminated = True
            info["reason"] = "Max turns reached"
        
        if abs(self.castle_control) >= 5:
            terminated = True
            if self.castle_control > 0:
                reward += 2.0
                info["reason"] = "Blue controls castle"
            else:
                reward -= 2.0
                info["reason"] = "Red controls castle"

        return self._get_obs(), reward, terminated, False, info

    def _advance_phase(self):
        if self.phase == "move":
            self.phase = "attack"
        else:
            self.phase = "move"
            self.unit_index_per_team[self.current_player] += 1
            team_units_alive = [u for u in self.units if u.team == self.current_player and u.is_alive()]
            if self.unit_index_per_team[self.current_player] >= len(team_units_alive):
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
            and pos not in self.castle_area
            and not any(u.position == pos and u.is_alive() for u in self.units)
        )

    def _get_reachable_cells(self, unit):
        reachable = np.zeros(self.board_size, dtype=np.float32)
        dirs = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]
        for dx, dy in dirs:
            new_pos = (unit.position[0] + dx, unit.position[1] + dy)
            if self._valid_move(new_pos):
                reachable[new_pos] = 1.0
        return reachable

    def _get_attackable_cells(self, unit):
        attackable = np.zeros(self.board_size, dtype=np.float32)
        attack_range = 3 if unit.unit_type == "Archer" else 1
        dirs = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]

        for dx, dy in dirs:
            for dist in range(1, attack_range + 1):
                tx, ty = unit.position[0] + dx * dist, unit.position[1] + dy * dist
                if not self._valid_coord((tx, ty)):
                    break

                if (tx, ty) in self.castle_area:
                    attackable[tx, ty] = 1.0
                else:
                    for enemy in self.units:
                        if enemy.team != self.current_player and enemy.is_alive() and enemy.position == (tx, ty):
                            attackable[tx, ty] = 1.0
                            break
            if attackable[tx, ty] == 1.0: # If target found in this direction, stop checking further distances
                break
        return attackable

    def _update_unit_actions_cache(self):
        self.cached_reachable_cells = np.zeros(self.board_size, dtype=np.float32)
        self.cached_attackable_cells = np.zeros(self.board_size, dtype=np.float32)

        team_units = [u for u in self.units if u.team == self.current_player and u.is_alive()]
        index = self.unit_index_per_team[self.current_player]

        if index < len(team_units):
            active_unit = team_units[index]
            if self.phase == "move":
                self.cached_reachable_cells = self._get_reachable_cells(active_unit)
            elif self.phase == "attack":
                self.cached_attackable_cells = self._get_attackable_cells(active_unit)

    def _get_obs(self):
        obs = np.zeros((self.num_obs_layers, *self.board_size), dtype=np.float32)
        
        obs[0] = self.obstacles
        for (x, y) in self.castle_area:
            obs[1, x, y] = 1.0
        
        team_units = [u for u in self.units if u.team == self.current_player and u.is_alive()]
        active_unit = None
        if self.unit_index_per_team[self.current_player] < len(team_units):
            active_unit = team_units[self.unit_index_per_team[self.current_player]]

        for unit in self.units:
            if not unit.is_alive():
                continue
            x, y = unit.position
            if unit.team == self.current_player:
                obs[2, x, y] = 1.0
                if unit.unit_type == "Soldier": obs[3, x, y] = 1.0
                elif unit.unit_type == "Knight": obs[4, x, y] = 1.0
                elif unit.unit_type == "Archer": obs[5, x, y] = 1.0
                obs[6, x, y] = unit.health / 100.0
            else:
                obs[7, x, y] = 1.0
                if unit.unit_type == "Soldier": obs[8, x, y] = 1.0
                elif unit.unit_type == "Knight": obs[9, x, y] = 1.0
                elif unit.unit_type == "Archer": obs[10, x, y] = 1.0
                obs[11, x, y] = unit.health / 100.0

            if active_unit and unit.position == active_unit.position and unit.team == active_unit.team:
                obs[12, x, y] = 1.0
                if active_unit.unit_type == "Soldier": obs[13, x, y] = 1.0
                elif active_unit.unit_type == "Knight": obs[14, x, y] = 1.0
                elif active_unit.unit_type == "Archer": obs[15, x, y] = 1.0
                obs[16, x, y] = active_unit.health / 100.0

        obs[17] = self.cached_reachable_cells
        obs[18] = self.cached_attackable_cells

        obs[19].fill(float(self.current_player))
        obs[20].fill(self.turn_count / self.max_turns)
        obs[21].fill((self.castle_control + 5) / 10.0)

        return obs

    def _is_adjacent_block_too_long(self, obstacles, x, y):
        if y >= 1 and y < obstacles.shape[1] - 1 and obstacles[x, y-1] == 1 and obstacles[x, y+1] == 1:
            return True
        if x >= 1 and x < obstacles.shape[0] - 1 and obstacles[x-1, y] == 1 and obstacles[x+1, y] == 1:
            return True
        return False

    def _generate_obstacles(self, units_positions, obstacle_count=10):
        attempts = 1000
        mid_x = self.board_size[0] // 2
        prohibited = set(self.castle_area)

        for x_row in [0, 1, self.board_size[0] - 2, self.board_size[0] - 1]:
            for y in range(self.board_size[1]):
                prohibited.add((x_row, y))

        for _ in range(attempts):
            obstacles = np.zeros(self.board_size, dtype=np.int8)
            placed = 0

            potential_positions = [
                (x, y)
                for x in range(2, mid_x)
                for y in range(self.board_size[1])
                if (x, y) not in prohibited and (x, y) not in units_positions
            ]
            random.shuffle(potential_positions)

            for x, y in potential_positions:
                mirror_x = self.board_size[0] - 1 - x
                mirror_pos = (mirror_x, y)

                if ((x, y) in units_positions or mirror_pos in units_positions or
                    (x, y) in prohibited or mirror_pos in prohibited):
                    continue
                
                obstacles[x, y] = 1
                if self._is_adjacent_block_too_long(obstacles, x, y):
                    obstacles[x, y] = 0
                    continue
                
                obstacles[mirror_x, y] = 1
                if self._is_adjacent_block_too_long(obstacles, mirror_x, y):
                    obstacles[x, y] = 0
                    obstacles[mirror_x, y] = 0
                    continue
                
                if placed + 2 <= obstacle_count:
                    placed += 2
                elif placed + 1 <= obstacle_count:
                    obstacles[mirror_x,y] = 0
                    placed += 1
                else:
                    obstacles[x,y] = 0
                    obstacles[mirror_x,y] = 0
                    continue

                if placed >= obstacle_count:
                    return obstacles

        print("⚠️ No se pudieron colocar todos los obstáculos deseados. Generando un tablero sin obstáculos o con menos.")
        return np.zeros(self.board_size, dtype=np.int8)