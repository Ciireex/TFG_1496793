import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
from gym_strategy.core.Unit import Soldier, Archer, Knight

class StrategyEnv_Base(gym.Env):
    def __init__(self, board_size=(4, 4), n_units_per_team=2, unit_types=[Soldier], max_turns=50):
        super().__init__()
        self.board_size = board_size
        self.max_turns = max_turns
        self.unit_types = unit_types
        self.n_units_per_team = n_units_per_team

        self.action_space = spaces.Discrete(5)  # [No-op, ↑, ↓, ←, →]
        self.observation_space = self._define_observation_space()

        self.reset()

    def _define_observation_space(self):
        n_channels = 5 + len(self.unit_types) * 2 + 8
        return spaces.Box(low=0, high=1, shape=(n_channels, *self.board_size), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._setup_game()
        return self._get_obs(), {}

    def _setup_game(self):
        self.turn_count = 0
        self.current_player = 0
        self.unit_index_per_team = {0: 0, 1: 0}
        self.phase = "move"
        self.units = []
        self.terrain = np.zeros(self.board_size, dtype=np.int8)

        all_coords = [(x, y) for x in range(self.board_size[0]) for y in range(self.board_size[1])]
        obstacle_coords = random.sample(all_coords, k=3)
        for x, y in obstacle_coords:
            self.terrain[x, y] = 99

        self._place_units()

    def _place_units(self):
        blue_positions = random.sample([(0, y) for y in range(self.board_size[1])], self.n_units_per_team)
        red_positions = random.sample([(self.board_size[0] - 1, y) for y in range(self.board_size[1])], self.n_units_per_team)
        for i in range(self.n_units_per_team):
            self.units.append(Soldier(position=blue_positions[i], team=0))
            self.units.append(Soldier(position=red_positions[i], team=1))

    def _get_obs(self):
        obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        unit_layer_offset = 5
        unit_layers = {}
        for idx, cls in enumerate(self.unit_types):
            unit_layers[(0, cls)] = unit_layer_offset + idx
            unit_layers[(1, cls)] = unit_layer_offset + idx + len(self.unit_types)

        # Obstáculos
        for x in range(self.board_size[0]):
            for y in range(self.board_size[1]):
                if self.terrain[x, y] == 99:
                    obs[0, x, y] = 1.0

        for unit in self.units:
            if unit.is_alive():
                x, y = unit.position
                obs[unit_layers[(unit.team, type(unit))], x, y] = 1.0
                health_layer = unit_layer_offset + 2 * len(self.unit_types)
                if unit.team != self.current_player:
                    health_layer += 1
                obs[health_layer, x, y] = unit.health / unit.max_health

        tactical_offset = unit_layer_offset + 2 * len(self.unit_types) + 2
        if active_unit := self._get_active_unit():
            x, y = active_unit.position
            obs[tactical_offset, x, y] = 1.0

            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                if self._valid_move((x + dx, y + dy)):
                    obs[tactical_offset + 1, x + dx, y + dy] = 1.0

            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                for dist in range(1, active_unit.attack_range + 1):
                    tx, ty = x + dx * dist, y + dy * dist
                    if self._valid_coord((tx, ty)):
                        obs[tactical_offset + 2, tx, ty] = 1.0

        obs[-3] = 1.0 if self.phase == "attack" else 0.0
        obs[-2] = self.turn_count / self.max_turns
        obs[-1] = self.current_player

        return obs

    def step(self, action):
        reward = 0
        terminated = False
        info = {}
        active_unit = self._get_active_unit()
        if not active_unit:
            self._advance_turn()
            return self._get_obs(), reward, terminated, False, info

        if self.phase == "move":
            dx, dy = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)][action]
            new_pos = (active_unit.position[0] + dx, active_unit.position[1] + dy)
            if action != 0 and self._valid_move(new_pos):
                active_unit.move(new_pos)
                reward += 0.05
            else:
                reward -= 0.01
            self.phase = "attack"

        else:
            attacked = False
            dx, dy = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)][action]
            for dist in range(1, active_unit.attack_range + 1):
                tx, ty = active_unit.position[0] + dx * dist, active_unit.position[1] + dy * dist
                if not self._valid_coord((tx, ty)):
                    break
                for enemy in [u for u in self.units if u.team != self.current_player and u.is_alive()]:
                    if enemy.position == (tx, ty):
                        damage = active_unit.get_attack_damage(enemy)
                        enemy.health -= damage
                        reward += 1.0
                        if isinstance(active_unit, Archer):
                            manhattan = abs(tx - active_unit.position[0]) + abs(ty - active_unit.position[1])
                            if manhattan in [2, 3]:
                                reward += 0.3
                        if not enemy.is_alive():
                            reward += 3.0
                            if self._team_won(self.current_player):
                                reward += 10.0
                                terminated = True
                        attacked = True
                        break
                if attacked:
                    break
            if not attacked and self._enemies_in_range(active_unit):
                reward -= 0.05
            self._advance_phase()

        if self.turn_count >= self.max_turns:
            terminated = True
            ally_alive = sum(1 for u in self.units if u.is_alive() and u.team == self.current_player)
            enemy_alive = sum(1 for u in self.units if u.is_alive() and u.team != self.current_player)
            if ally_alive > enemy_alive:
                reward += 5.0
            elif ally_alive < enemy_alive:
                reward -= 5.0
            else:
                reward -= 2.0

        return self._get_obs(), reward, terminated, False, info

    def _get_active_unit(self):
        team_units = [u for u in self.units if u.team == self.current_player and u.is_alive()]
        if not team_units:
            return None
        return team_units[self.unit_index_per_team[self.current_player] % len(team_units)]

    def _valid_coord(self, pos):
        x, y = pos
        return 0 <= x < self.board_size[0] and 0 <= y < self.board_size[1]

    def _valid_move(self, pos):
        return (
            self._valid_coord(pos) and
            self.terrain[pos[0], pos[1]] != 99 and
            not any(u.position == pos and u.is_alive() for u in self.units)
        )

    def _enemies_in_range(self, unit):
        x, y = unit.position
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            for dist in range(1, unit.attack_range + 1):
                tx, ty = x + dx * dist, y + dy * dist
                if not self._valid_coord((tx, ty)):
                    break
                for enemy in [u for u in self.units if u.team != unit.team and u.is_alive()]:
                    if enemy.position == (tx, ty):
                        return True
        return False

    def _advance_phase(self):
        if self.phase == "move":
            self.phase = "attack"
        else:
            self.phase = "move"
            self.unit_index_per_team[self.current_player] += 1
            team_units = [u for u in self.units if u.team == self.current_player and u.is_alive()]
            if self.unit_index_per_team[self.current_player] >= len(team_units):
                self.unit_index_per_team[self.current_player] = 0
                self.current_player = 1 - self.current_player
                self.turn_count += 1

    def _team_won(self, team):
        return all(not u.is_alive() for u in self.units if u.team != team)

class StrategyEnv_2v2Soldiers4x4(StrategyEnv_Base):
    def __init__(self):
        super().__init__(board_size=(4, 4), n_units_per_team=2, unit_types=[Soldier], max_turns=50)

class StrategyEnv_3v3Soldiers6x4(StrategyEnv_Base):
    def __init__(self):
        super().__init__(board_size=(6, 4), n_units_per_team=3, unit_types=[Soldier], max_turns=60)

class StrategyEnv_3v3Soldiers6x4_Obs(StrategyEnv_Base):
    def __init__(self):
        super().__init__(board_size=(6, 4), n_units_per_team=3, unit_types=[Soldier], max_turns=60)

    def _setup_game(self):
        self.turn_count = 0
        self.current_player = 0
        self.unit_index_per_team = {0: 0, 1: 0}
        self.phase = "move"
        self.units = []
        self.terrain = np.zeros(self.board_size, dtype=np.int8)

        # Añadir 3 obstáculos aleatorios evitando esquinas iniciales
        forbidden = set([(0, y) for y in range(self.board_size[1])] + [(self.board_size[0]-1, y) for y in range(self.board_size[1])])
        all_coords = [(x, y) for x in range(self.board_size[0]) for y in range(self.board_size[1]) if (x, y) not in forbidden]
        obstacle_coords = random.sample(all_coords, k=3)
        for x, y in obstacle_coords:
            self.terrain[x, y] = 99

        self._place_units()

class StrategyEnv_2Soldiers1Archer_6x4_Obs(StrategyEnv_Base):
    def __init__(self):
        super().__init__(board_size=(6, 4), n_units_per_team=3, unit_types=[Soldier, Archer], max_turns=60)

    def _place_units(self):
        blue_types = [Soldier, Soldier, Archer]
        red_types = [Soldier, Soldier, Archer]

        blue_positions = random.sample([(0, y) for y in range(self.board_size[1])], 3)
        red_positions = random.sample([(self.board_size[0] - 1, y) for y in range(self.board_size[1])], 3)

        for i in range(3):
            self.units.append(blue_types[i](position=blue_positions[i], team=0))
            self.units.append(red_types[i](position=red_positions[i], team=1))

    def _setup_game(self):
        self.turn_count = 0
        self.current_player = 0
        self.unit_index_per_team = {0: 0, 1: 0}
        self.phase = "move"
        self.units = []
        self.terrain = np.zeros(self.board_size, dtype=np.int8)

        forbidden = set([(0, y) for y in range(self.board_size[1])] + [(self.board_size[0] - 1, y) for y in range(self.board_size[1])])
        all_coords = [(x, y) for x in range(self.board_size[0]) for y in range(self.board_size[1]) if (x, y) not in forbidden]
        obstacle_coords = random.sample(all_coords, k=3)
        for x, y in obstacle_coords:
            self.terrain[x, y] = 99

        self._place_units()

class StrategyEnv_3Units_6x4(gym.Env):
    def __init__(self):
        super().__init__()
        self.board_size = (6, 4)
        self.max_turns = 60
        self.current_turn = 0

        self.unit_types = [Soldier, Archer, Knight]
        self.units = []
        self.current_player = 0  # 0: Blue, 1: Red
        self.unit_index = 0

        self.observation_space = spaces.Box(low=0, high=1, shape=(16, 6, 4), dtype=np.float32)
        self.action_space = spaces.MultiDiscrete([5, 5])  # Move: 0-4, Attack: 0-4

        self.seed()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.units = []
        self.current_turn = 0
        self.current_player = 0
        self.unit_index = 0

        # Posiciones iniciales para 1 Soldier, 1 Archer y 1 Knight por equipo
        self.units.append(Soldier(position=(0, 1), team=0))
        self.units.append(Archer(position=(0, 2), team=0))
        self.units.append(Knight(position=(0, 3), team=0))

        self.units.append(Soldier(position=(5, 1), team=1))
        self.units.append(Archer(position=(5, 2), team=1))
        self.units.append(Knight(position=(5, 3), team=1))

        return self._get_obs(), {}

    def step(self, action):
        move_dir, atk_dir = action
        unit = self._get_active_unit()

        # Movimiento
        new_pos = self._get_target_position(unit.position, move_dir)
        if self._is_valid_position(new_pos):
            if not self._unit_at(new_pos):
                unit.move(new_pos)

        # Ataque
        target_pos = self._get_target_position(unit.position, atk_dir)
        target_unit = self._unit_at(target_pos, enemy_only=True)
        unit.attack(target_unit, self)  # `self` es usado por Knight

        # Siguiente turno
        done = self._next_unit()
        reward = self._compute_reward()

        return self._get_obs(), reward, done, False, {}

    def _get_obs(self):
        obs = np.zeros((16, 6, 4), dtype=np.float32)
        for unit in self.units:
            x, y = unit.position
            if unit.team == 0:
                if unit.unit_type == "Soldier": obs[0, x, y] = 1
                if unit.unit_type == "Archer": obs[1, x, y] = 1
                if unit.unit_type == "Knight": obs[2, x, y] = 1
            else:
                if unit.unit_type == "Soldier": obs[3, x, y] = 1
                if unit.unit_type == "Archer": obs[4, x, y] = 1
                if unit.unit_type == "Knight": obs[5, x, y] = 1
            obs[6, x, y] = unit.health / 100
        # Activa
        ux, uy = self._get_active_unit().position
        obs[7, ux, uy] = 1
        return obs

    def _next_unit(self):
        self.unit_index += 1
        while self.unit_index < len(self.units):
            if self.units[self.unit_index].team == self.current_player and self.units[self.unit_index].is_alive():
                return False
            self.unit_index += 1

        # Cambio de jugador
        self.current_player = 1 - self.current_player
        self.unit_index = 0
        self.current_turn += 1

        # Verificar final de partida
        if self.current_turn >= self.max_turns:
            return True
        if not any(u.team == 0 and u.is_alive() for u in self.units):
            return True
        if not any(u.team == 1 and u.is_alive() for u in self.units):
            return True

        return False

    def _get_target_position(self, position, direction):
        x, y = position
        if direction == 1: return (x-1, y)  # Arriba
        if direction == 2: return (x+1, y)  # Abajo
        if direction == 3: return (x, y-1)  # Izquierda
        if direction == 4: return (x, y+1)  # Derecha
        return (x, y)

    def _is_valid_position(self, pos):
        x, y = pos
        return 0 <= x < self.board_size[0] and 0 <= y < self.board_size[1]

    def _unit_at(self, pos, enemy_only=False):
        for unit in self.units:
            if unit.position == pos and unit.is_alive():
                if enemy_only and unit.team == self.current_player:
                    continue
                return unit
        return None

    def _get_active_unit(self):
        for i in range(self.unit_index, len(self.units)):
            if self.units[i].team == self.current_player and self.units[i].is_alive():
                return self.units[i]
        # En caso de error
        return random.choice([u for u in self.units if u.team == self.current_player and u.is_alive()])

    def is_within_bounds(self, pos):
        return self._is_valid_position(pos)

    def is_empty(self, pos):
        return self._unit_at(pos) is None

    def _compute_reward(self):
        return 0  # Ajusta esto según el objetivo de entrenamiento
    
class StrategyEnv_Knight(StrategyEnv_Base):
    def __init__(self):
        super().__init__(board_size=(6, 4), n_units_per_team=4, unit_types=[Soldier, Archer, Knight], max_turns=70)

    def _place_units(self):
        blue_types = [Soldier, Soldier, Archer, Knight]
        red_types = [Soldier, Soldier, Archer, Knight]

        blue_positions = random.sample([(0, y) for y in range(self.board_size[1])] + [(1, y) for y in range(self.board_size[1])], 4)
        red_positions = random.sample([(self.board_size[0]-1, y) for y in range(self.board_size[1])] + [(self.board_size[0]-2, y) for y in range(self.board_size[1])], 4)

        for i in range(4):
            self.units.append(blue_types[i](position=blue_positions[i], team=0))
            self.units.append(red_types[i](position=red_positions[i], team=1))

    def step(self, action):
        reward = 0
        terminated = False
        info = {}
        active_unit = self._get_active_unit()
        if not active_unit:
            self._advance_turn()
            return self._get_obs(), reward, terminated, False, info

        if self.phase == "move":
            dx, dy = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)][action]
            new_pos = (active_unit.position[0] + dx, active_unit.position[1] + dy)
            if action != 0 and self._valid_move(new_pos):
                active_unit.move(new_pos)
                reward += 0.05
            else:
                reward -= 0.01
            self.phase = "attack"

        else:
            attacked = False
            dx, dy = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)][action]
            for dist in range(1, active_unit.attack_range + 1):
                tx, ty = active_unit.position[0] + dx * dist, active_unit.position[1] + dy * dist
                if not self._valid_coord((tx, ty)):
                    break
                for enemy in [u for u in self.units if u.team != self.current_player and u.is_alive()]:
                    if enemy.position == (tx, ty):
                        prev_health = enemy.health
                        prev_pos = enemy.position
                        enemy_prev_alive = enemy.is_alive()

                        if isinstance(active_unit, Knight):
                            dx_e = enemy.position[0] - active_unit.position[0]
                            dy_e = enemy.position[1] - active_unit.position[1]
                            push_pos = (enemy.position[0] + dx_e, enemy.position[1] + dy_e)
                            can_push = (
                                self._valid_coord(push_pos) and
                                self._valid_move(push_pos) and
                                not any(u.position == push_pos and u.is_alive() for u in self.units)
                            )
                            active_unit.attack(enemy, self)
                            if can_push:
                                reward += 0.5
                            else:
                                reward += 0.8  # Más daño si no puede empujar
                        else:
                            active_unit.attack(enemy)

                        reward += 1.0
                        if isinstance(active_unit, Archer):
                            manhattan = abs(tx - active_unit.position[0]) + abs(ty - active_unit.position[1])
                            if manhattan in [2, 3]:
                                reward += 0.3

                        if not enemy.is_alive() and enemy_prev_alive:
                            reward += 3.0
                            if self._team_won(self.current_player):
                                reward += 10.0
                                terminated = True

                        attacked = True
                        break
                if attacked:
                    break

            if not attacked and self._enemies_in_range(active_unit):
                reward -= 0.05
            self._advance_phase()

        if self.turn_count >= self.max_turns:
            terminated = True
            ally_alive = sum(1 for u in self.units if u.is_alive() and u.team == self.current_player)
            enemy_alive = sum(1 for u in self.units if u.is_alive() and u.team != self.current_player)
            if ally_alive > enemy_alive:
                reward += 5.0
            elif ally_alive < enemy_alive:
                reward -= 5.0
            else:
                reward -= 2.0

        return self._get_obs(), reward, terminated, False, info
