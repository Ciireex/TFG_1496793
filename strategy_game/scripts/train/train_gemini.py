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
import sys

# --- Custom CNN Feature Extractor ---
class CustomCNN(BaseFeaturesExtractor):
    """
    Custom CNN feature extractor for the StrategyEnv_Gemini observation space.
    Processes the multi-channel grid observation into a feature vector.
    """
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
            # observation_space.sample() gives a sample observation. [None] adds batch dimension.
            sample_input = torch.as_tensor(observation_space.sample()[None]).float()
            n_flatten = self.cnn(sample_input).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
            nn.Dropout(p=0.2)  # Soft regularization for better generalization
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the CNN and linear layers.
        """
        return self.linear(self.cnn(observations))

# --- Unit Classes ---
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
    def __init__(self, position, team): # <-- Corrected: 'self' is present
        super().__init__(position, team, "Archer", health=70, attack_damage=10)

class Knight(Unit):
    def __init__(self, position, team):
        super().__init__(position, team, "Knight", health=120, attack_damage=20)

# --- StrategyEnv_Gemini Environment ---
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
            target_found_in_direction = False # Flag to stop checking further distances in this direction
            for dist in range(1, attack_range + 1):
                tx, ty = unit.position[0] + dx * dist, unit.position[1] + dy * dist
                
                # Check bounds BEFORE trying to access array
                if not self._valid_coord((tx, ty)):
                    break # Stop checking this direction if out of bounds

                # Check for castle
                if (tx, ty) in self.castle_area:
                    attackable[tx, ty] = 1.0
                    target_found_in_direction = True
                    break # Target found, stop checking further distances in this direction
                
                # Check for enemy units
                for enemy in self.units:
                    if enemy.team != self.current_player and enemy.is_alive() and enemy.position == (tx, ty):
                        attackable[tx, ty] = 1.0
                        target_found_in_direction = True
                        break # Target found, stop checking further distances in this direction
                
                if target_found_in_direction: # If a target was found by the inner loop, break the outer dist loop
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


# --- OpponentModelWrapper ---
class OpponentModelWrapper(gym.Wrapper):
    """
    A Gym wrapper to integrate an opponent model into the environment.
    This allows a single-agent RL algorithm to train against an opponent,
    which can be another RL agent (for self-play) or a rule-based agent.
    """
    def __init__(self, env, opponent_model: BaseAlgorithm = None, training_agent_team: int = 0):
        super().__init__(env)
        self.opponent_model = opponent_model
        self.training_agent_team = training_agent_team # The team the PPO agent in the outer loop is training

    def step(self, action):
        # The training agent (self.training_agent_team) takes its action
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Let the opponent act until it's the training agent's turn again, or the game ends
        while self.env.current_player != self.training_agent_team and not (terminated or truncated):
            if self.opponent_model:
                # The environment's `_get_obs()` method automatically returns the observation
                # from the perspective of the `self.env.current_player`. So, `obs` here
                # is already the correct observation for the opponent model.
                opponent_action, _ = self.opponent_model.predict(obs, deterministic=False)
                obs, opponent_reward, terminated, truncated, info = self.env.step(opponent_action)
            else:
                # If no opponent model, the opponent just takes a "stay" action.
                obs, opponent_reward, terminated, truncated, info = self.env.step(0) # Action 0 = stay

        # Return the observation from the perspective of the *training agent*
        # (after the opponent has finished its turn(s)).
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # If the training agent is team 1, the first turn will be team 0 (opponent).
        # Let the opponent act until it's the training agent's turn.
        while self.env.current_player != self.training_agent_team:
            if self.opponent_model:
                # The initial `obs` from `env.reset()` is from Team 0's perspective.
                # If training agent is Team 1, this `obs` *is* the opponent's observation.
                opponent_action, _ = self.opponent_model.predict(obs, deterministic=False)
                obs, opponent_reward, terminated, truncated, info = self.env.step(opponent_action)
                if terminated or truncated:
                    # If game ends before training agent's first turn, reset and try again
                    obs, info = self.env.reset(**kwargs)
            else:
                # If no opponent model, opponent takes dummy actions
                obs, _, terminated, truncated, _ = self.env.step(0)
                if terminated or truncated:
                    obs, info = self.env.reset(**kwargs)
        return obs, info

# --- Helper for Evaluation ---
def evaluate_model(model, env_factory, opponent_model=None, training_agent_team=0, num_episodes=10, render=False):
    """
    Evaluates a trained model in the given environment.
    Returns the mean reward and standard deviation.
    The env_factory is a callable that returns a new environment instance.
    """
    env = OpponentModelWrapper(env_factory(), opponent_model=opponent_model, training_agent_team=training_agent_team)
    episode_rewards = []
    for i in range(num_episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0
        while not done:
            action, _states = model.predict(obs, deterministic=True) # Use deterministic policy for evaluation
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            if render:
                env.render()
        episode_rewards.append(episode_reward)
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    env.close() # Close the evaluation environment after use
    return mean_reward, std_reward

# --- Self-Play Training Loop ---
def train_self_play(env_class, total_training_steps_per_cycle, num_cycles, save_dir="self_play_models_custom_cnn"):
    """
    Implements a self-play training loop for two agents in a competitive environment.
    """
    os.makedirs(save_dir, exist_ok=True)

    # Define policy_kwargs to use CustomCNN as the feature extractor
    policy_kwargs = dict(
        features_extractor_class=CustomCNN,
        features_extractor_kwargs=dict(features_dim=256), # Customize features_dim as needed
    )

    print("Inicializando modelos para Equipo 0 y Equipo 1 con CustomCNN...")
    model_team0_initial_path = os.path.join(save_dir, "model_team0_initial.zip")
    model_team1_initial_path = os.path.join(save_dir, "model_team1_initial.zip")

    # Initialize or load initial models
    if not os.path.exists(model_team0_initial_path):
        model_team0 = PPO(ActorCriticCnnPolicy, env_class(), verbose=0,
                          n_steps=512, batch_size=64, gamma=0.99, ent_coef=0.01,
                          learning_rate=0.0003, device="auto", policy_kwargs=policy_kwargs)
        model_team0.save(model_team0_initial_path)
    else:
        print(f"Cargando modelo inicial existente para Equipo 0: {model_team0_initial_path}")
        model_team0 = PPO.load(model_team0_initial_path, env=None, custom_objects={"CustomCNN": CustomCNN})


    if not os.path.exists(model_team1_initial_path):
        model_team1 = PPO(ActorCriticCnnPolicy, env_class(), verbose=0,
                          n_steps=512, batch_size=64, gamma=0.99, ent_coef=0.01,
                          learning_rate=0.0003, device="auto", policy_kwargs=policy_kwargs)
        model_team1.save(model_team1_initial_path)
    else:
        print(f"Cargando modelo inicial existente para Equipo 1: {model_team1_initial_path}")
        model_team1 = PPO.load(model_team1_initial_path, env=None, custom_objects={"CustomCNN": CustomCNN})


    # Track best rewards for saving the best performing models
    best_reward_team0 = -np.inf
    best_reward_team1 = -np.inf

    for cycle in range(num_cycles):
        print(f"\n--- Ciclo {cycle + 1}/{num_cycles} ---")

        # --- Fase 1: Entrenar al Agente del Equipo 0 (vs. Agente actual del Equipo 1) ---
        print(f"Entrenando Equipo 0 (Agente Azul) contra Equipo 1 (Agente Rojo actual)...")
        
        # Load the latest opponent model (Team 1's model from the PREVIOUS cycle's training)
        opponent_team1_model_path = os.path.join(save_dir, f"model_team1_cycle_{cycle-1}.zip") if cycle > 0 else model_team1_initial_path
        current_opponent_team1 = PPO.load(opponent_team1_model_path, env=None, custom_objects={"CustomCNN": CustomCNN})
        
        # Create a factory function for make_vec_env to produce wrapped environments
        env_team0_train_factory = lambda: OpponentModelWrapper(env_class(),
                                                              opponent_model=current_opponent_team1,
                                                              training_agent_team=0)
        vec_env_team0 = make_vec_env(env_team0_train_factory, n_envs=1) # Use n_envs=1 for simplicity, can be increased

        # Load the model for Team 0 to continue training.
        # This should always load the model from the *previous* cycle's training *for this specific team*.
        model_team0_to_load_path = os.path.join(save_dir, f"model_team0_cycle_{cycle-1}.zip") if cycle > 0 else model_team0_initial_path
        model_team0 = PPO.load(model_team0_to_load_path, env=vec_env_team0, custom_objects={"CustomCNN": CustomCNN})
        model_team0.set_env(vec_env_team0) # Ensure model uses the new wrapped env

        # Train Team 0
        model_team0.learn(total_timesteps=total_training_steps_per_cycle, reset_num_timesteps=False)
        model_team0.save(os.path.join(save_dir, f"model_team0_cycle_{cycle}.zip")) # Save as current cycle number
        print(f"Equipo 0 entrenado. Guardado como model_team0_cycle_{cycle}.zip")

        # --- Evaluación del Equipo 0 ---
        # Evaluate Team 0 against the current opponent (Team 1)
        mean_reward_team0, _ = evaluate_model(model_team0, env_class,
                                             opponent_model=current_opponent_team1,
                                             training_agent_team=0)
        print(f"Recompensa promedio del Equipo 0 en Ciclo {cycle+1}: {mean_reward_team0:.2f}")
        if mean_reward_team0 > best_reward_team0:
            best_reward_team0 = mean_reward_team0
            model_team0.save(os.path.join(save_dir, "best_model_team0.zip"))
            print(f"¡Nueva mejor recompensa para Equipo 0! Guardado como best_model_team0.zip")


        # --- Fase 2: Entrenar al Agente del Equipo 1 (vs. Agente recién entrenado del Equipo 0) ---
        print(f"Entrenando Equipo 1 (Agente Rojo) contra Equipo 0 (Agente Azul recién entrenado)...")
        # Load the latest opponent model (Team 0's model, which was just trained in THIS cycle)
        opponent_team0_model_path = os.path.join(save_dir, f"model_team0_cycle_{cycle}.zip") # Use current cycle for opponent
        current_opponent_team0 = PPO.load(opponent_team0_model_path, env=None, custom_objects={"CustomCNN": CustomCNN})

        # Create environment for Team 1 training
        env_team1_train_factory = lambda: OpponentModelWrapper(env_class(),
                                                              opponent_model=current_opponent_team0,
                                                              training_agent_team=1) # Training Team 1
        vec_env_team1 = make_vec_env(env_team1_train_factory, n_envs=1)

        # Load the model for Team 1 to continue training.
        # For cycle 0, it loads the initial. For subsequent cycles, it loads from 'cycle-1'.
        model_team1_to_load_path = os.path.join(save_dir, f"model_team1_cycle_{cycle-1}.zip") if cycle > 0 else model_team1_initial_path
        model_team1 = PPO.load(model_team1_to_load_path, env=vec_env_team1, custom_objects={"CustomCNN": CustomCNN})
        model_team1.set_env(vec_env_team1) # Ensure model uses the new wrapped env

        # Train Team 1
        model_team1.learn(total_timesteps=total_training_steps_per_cycle, reset_num_timesteps=False)
        model_team1.save(os.path.join(save_dir, f"model_team1_cycle_{cycle}.zip")) # Save as current cycle number
        print(f"Equipo 1 entrenado. Guardado como model_team1_cycle_{cycle}.zip")

        # --- Evaluación del Equipo 1 ---
        # Evaluate Team 1 against the current opponent (Team 0)
        mean_reward_team1, _ = evaluate_model(model_team1, env_class,
                                             opponent_model=current_opponent_team0,
                                             training_agent_team=1)
        print(f"Recompensa promedio del Equipo 1 en Ciclo {cycle+1}: {mean_reward_team1:.2f}")
        if mean_reward_team1 > best_reward_team1:
            best_reward_team1 = mean_reward_team1
            model_team1.save(os.path.join(save_dir, "best_model_team1.zip"))
            print(f"¡Nueva mejor recompensa para Equipo 1! Guardado como best_model_team1.zip")

    print("\n¡Entrenamiento por ciclos completado!")
    print(f"Mejor recompensa Equipo 0: {best_reward_team0:.2f}")
    print(f"Mejor recompensa Equipo 1: {best_reward_team1:.2f}")


if __name__ == '__main__':
    # Set training parameters
    TOTAL_TRAINING_STEPS_PER_CYCLE = 50_000 # Number of steps each agent trains per cycle
    NUM_CYCLES = 10 # Number of times both agents train against each other

    # Start the self-play training process
    train_self_play(StrategyEnv_Gemini, TOTAL_TRAINING_STEPS_PER_CYCLE, NUM_CYCLES)

    # --- After training, you can load and test the best models ---
    print("\nCargando los mejores modelos entrenados para jugar entre ellos...")
    save_directory = "self_play_models_custom_cnn" # Updated save_dir for consistency
    try:
        # Load the best models, specifying the custom_objects
        best_model_team0 = PPO.load(os.path.join(save_directory, "best_model_team0.zip"), custom_objects={"CustomCNN": CustomCNN})
        best_model_team1 = PPO.load(os.path.join(save_directory, "best_model_team1.zip"), custom_objects={"CustomCNN": CustomCNN})

        # Set up an environment for a match between the best models
        # We simulate from Team 0's perspective, with Team 1 as the opponent.
        final_env_team0_perspective = OpponentModelWrapper(StrategyEnv_Gemini(),
                                                          opponent_model=best_model_team1,
                                                          training_agent_team=0) # Team 0 is the "training agent" for this final run

        print("\nSimulando un juego entre los mejores modelos:")
        obs, info = final_env_team0_perspective.reset()
        done = False
        episode_reward_team0 = 0
        turns_played = 0

        # Run the simulation
        while not done and turns_played < StrategyEnv_Gemini().max_turns:
            # Team 0 (the current `training_agent_team` for this wrapper) acts
            action_team0, _ = best_model_team0.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = final_env_team0_perspective.step(action_team0)
            done = terminated or truncated
            episode_reward_team0 += reward
            turns_played += 1 # Each `step` in the wrapper represents a full round of turns (agent + opponent)

            if done:
                break
        
        final_control = final_env_team0_perspective.unwrapped.castle_control # Get final castle control from the base env
        print(f"Juego finalizado. Recompensa del Equipo 0 (Azul): {episode_reward_team0:.2f}")
        if final_control > 0:
            print(f"Resultado: ¡El Equipo 0 (Azul) controla el castillo con {final_control} puntos de control!")
        elif final_control < 0:
            print(f"Resultado: ¡El Equipo 1 (Rojo) controla el castillo con {abs(final_control)} puntos de control!")
        else:
            print("Resultado: Empate o tiempo agotado sin control decisivo del castillo.")

        final_env_team0_perspective.close()

    except FileNotFoundError:
        print(f"No se encontraron modelos guardados en '{save_directory}'. Por favor, ejecuta el entrenamiento primero.")