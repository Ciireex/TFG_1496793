import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
from gym_strategy.core.Unit import Soldier, Archer, Knight

class StrategyEnv_Def(gym.Env):
    def __init__(self, board_size=(4, 4), unit_types=[Soldier], team_composition=None, max_turns=50, n_obstacles=0):
        super().__init__()
        self.board_size = board_size
        self.unit_types = unit_types
        self.team_composition = team_composition or [Soldier] * 2
        self.n_units_per_team = len(self.team_composition)
        self.max_turns = max_turns
        self.n_obstacles = n_obstacles

        self.action_space = spaces.Discrete(5)
        self.observation_space = self._define_observation_space()

        self.reset()

    def _define_observation_space(self):
        n_unit_types = len([Soldier, Archer, Knight])  # Siempre reservar canales para los 3
        n_channels = 5 + 2 * n_unit_types + 2 + 5 + 3  # Obstáculos + unidades + vida + táctica + fase + terreno
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

        if self.n_obstacles > 0:
            forbidden = set([(0, y) for y in range(self.board_size[1])] + [(self.board_size[0]-1, y) for y in range(self.board_size[1])])
            all_coords = [(x, y) for x in range(self.board_size[0]) for y in range(self.board_size[1]) if (x, y) not in forbidden]
            obstacle_coords = random.sample(all_coords, k=self.n_obstacles)
            for x, y in obstacle_coords:
                self.terrain[x, y] = 99

        self._place_units()

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
                        prev_alive = enemy.is_alive()

                        if isinstance(active_unit, Knight):
                            dx_e = enemy.position[0] - active_unit.position[0]
                            dy_e = enemy.position[1] - active_unit.position[1]
                            push_pos = (enemy.position[0] + dx_e, enemy.position[1] + dy_e)
                            can_push = (
                                self._valid_coord(push_pos)
                                and self._valid_move(push_pos)
                                and not any(u.position == push_pos and u.is_alive() for u in self.units)
                            )
                            active_unit.attack(enemy, self)
                            reward += 0.5 if can_push else 0.8
                        else:
                            active_unit.attack(enemy)

                        reward += 1.0

                        if isinstance(active_unit, Archer):
                            manhattan = abs(tx - active_unit.position[0]) + abs(ty - active_unit.position[1])
                            if manhattan in [2, 3]:
                                reward += 0.3

                        if not enemy.is_alive() and prev_alive:
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

        # 👇 Penalización por llegar al máximo de turnos sin terminar
        if self.turn_count >= self.max_turns:
            terminated = True

            ally_alive = sum(1 for u in self.units if u.is_alive() and u.team == self.current_player)
            enemy_alive = sum(1 for u in self.units if u.is_alive() and u.team != self.current_player)

            reward -= 3.0  # penalización general por no acabar

            if enemy_alive == 0:
                reward += 10.0  # victoria
            elif ally_alive == 0:
                reward -= 10.0  # derrota
            elif ally_alive > enemy_alive:
                reward += 1.0  # ventaja
            elif ally_alive < enemy_alive:
                reward -= 1.0  # desventaja
            else:
                reward -= 2.0  # empate

        return self._get_obs(), reward, terminated, False, info

    def step2(self, action):
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
                        prev_alive = enemy.is_alive()
                        prev_pos = enemy.position

                        if isinstance(active_unit, Knight):
                            dx_e = enemy.position[0] - active_unit.position[0]
                            dy_e = enemy.position[1] - active_unit.position[1]
                            push_pos = (enemy.position[0] + dx_e, enemy.position[1] + dy_e)
                            can_push = (
                                self._valid_coord(push_pos)
                                and self._valid_move(push_pos)
                                and not any(u.position == push_pos and u.is_alive() for u in self.units)
                            )
                            active_unit.attack(enemy, self)
                            reward += 0.5 if can_push else 0.8
                        else:
                            active_unit.attack(enemy)

                        reward += 1.0

                        if isinstance(active_unit, Archer):
                            manhattan = abs(tx - active_unit.position[0]) + abs(ty - active_unit.position[1])
                            if manhattan in [2, 3]:
                                reward += 0.3

                        if not enemy.is_alive() and prev_alive:
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
    
    def _place_units(self):
        # Filtrar solo posiciones sin obstáculos
        def get_valid_positions(rows):
            return [
                (x, y)
                for x in rows
                for y in range(self.board_size[1])
                if self.terrain[x, y] != 99
            ]

        # Obtener posiciones válidas
        blue_candidates = get_valid_positions([0, 1])
        red_candidates = get_valid_positions([self.board_size[0]-2, self.board_size[0]-1])

        # Comprobar que hay suficientes
        if len(blue_candidates) < self.n_units_per_team or len(red_candidates) < self.n_units_per_team:
            raise ValueError("No hay suficientes posiciones libres para colocar las unidades")

        # Elegir posiciones aleatorias
        blue_positions = random.sample(blue_candidates, self.n_units_per_team)
        red_positions = random.sample(red_candidates, self.n_units_per_team)

        for i in range(self.n_units_per_team):
            blue_cls = self.team_composition[i]
            red_cls = self.team_composition[i]
            self.units.append(blue_cls(position=blue_positions[i], team=0))
            self.units.append(red_cls(position=red_positions[i], team=1))


    def _place_units2(self):
        blue_positions = random.sample([(0, y) for y in range(self.board_size[1])] + [(1, y) for y in range(self.board_size[1])], self.n_units_per_team)
        red_positions = random.sample([(self.board_size[0]-1, y) for y in range(self.board_size[1])] + [(self.board_size[0]-2, y) for y in range(self.board_size[1])], self.n_units_per_team)

        for i in range(self.n_units_per_team):
            blue_cls = self.team_composition[i]
            red_cls = self.team_composition[i]
            self.units.append(blue_cls(position=blue_positions[i], team=0))
            self.units.append(red_cls(position=red_positions[i], team=1))

    def _get_obs(self):
        obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        unit_layer_offset = 1
        unit_map = {Soldier: 0, Archer: 1, Knight: 2}

        for x in range(self.board_size[0]):
            for y in range(self.board_size[1]):
                if self.terrain[x, y] == 99:
                    obs[0, x, y] = 1.0

        for unit in self.units:
            if not unit.is_alive():
                continue
            x, y = unit.position
            idx = unit_map[type(unit)]
            if unit.team == 0:
                obs[unit_layer_offset + idx, x, y] = 1.0
                obs[unit_layer_offset + 3 + idx, x, y] = unit.health / unit.max_health
            else:
                obs[unit_layer_offset + 6 + idx, x, y] = 1.0
                obs[unit_layer_offset + 9 + idx, x, y] = unit.health / unit.max_health

        # Unidad activa
        tactical_offset = unit_layer_offset + 12
        if active_unit := self._get_active_unit():
            x, y = active_unit.position
            obs[tactical_offset, x, y] = 1.0
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dx, y + dy
                if self._valid_move((nx, ny)):
                    obs[tactical_offset + 1, nx, ny] = 1.0
                for dist in range(1, active_unit.attack_range + 1):
                    tx, ty = x + dx * dist, y + dy * dist
                    if self._valid_coord((tx, ty)):
                        obs[tactical_offset + 2, tx, ty] = 1.0

        obs[-3] = 1.0 if self.phase == "attack" else 0.0
        obs[-2] = self.turn_count / self.max_turns
        obs[-1] = self.current_player

        return obs
    
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
    
    def _team_won(self, team):
        return all(not u.is_alive() for u in self.units if u.team != team)

    def _valid_coord(self, pos):
        x, y = pos
        return 0 <= x < self.board_size[0] and 0 <= y < self.board_size[1]

    def _valid_move(self, pos):
        return (
            self._valid_coord(pos) and
            self.terrain[pos[0], pos[1]] != 99 and
            not any(u.position == pos and u.is_alive() for u in self.units)
        )

    def _get_active_unit(self):
        team_units = [u for u in self.units if u.team == self.current_player and u.is_alive()]
        if not team_units:
            return None
        return team_units[self.unit_index_per_team[self.current_player] % len(team_units)]

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

    def is_within_bounds(self, pos):
        return self._valid_coord(pos)
    
    def is_empty(self, pos):
        return (
            self._valid_coord(pos)
            and self.terrain[pos[0], pos[1]] != 99
            and not any(u.position == pos and u.is_alive() for u in self.units)
        )

# Ejemplos para cada fase de entrenamiento:
class Env_Fase1_Soldiers4x4(StrategyEnv_Def):
    def __init__(self):
        super().__init__(board_size=(4, 4), team_composition=[Soldier, Soldier], unit_types=[Soldier], max_turns=50)

class Env_Fase2_Soldiers6x4(StrategyEnv_Def):
    def __init__(self):
        super().__init__(board_size=(6, 4), team_composition=[Soldier] * 3, unit_types=[Soldier], max_turns=60)

class Env_Fase3_Obstaculos(StrategyEnv_Def):
    def __init__(self):
        super().__init__(board_size=(6, 4), team_composition=[Soldier] * 3, unit_types=[Soldier], max_turns=60, n_obstacles=3)

class Env_Fase4_Archer(StrategyEnv_Def):
    def __init__(self):
        super().__init__(board_size=(6, 4), team_composition=[Soldier, Soldier, Archer], unit_types=[Soldier, Archer], max_turns=60, n_obstacles=3)

class Env_Fase5_Knight(StrategyEnv_Def):
    def __init__(self):
        super().__init__(board_size=(6, 4), team_composition=[Soldier, Soldier, Archer, Knight], unit_types=[Soldier, Archer, Knight], max_turns=70, n_obstacles=3)

class Env_Fase6_MapaGrande(StrategyEnv_Def):
    def __init__(self):
        super().__init__(
            board_size=(10, 6),
            team_composition=[Soldier, Soldier, Archer, Archer, Knight, Knight],
            unit_types=[Soldier, Archer, Knight],
            max_turns=80,
            n_obstacles=6  # puedes ajustar este número si quieres más complejidad
        )

class Env_Fase7_Terreno(StrategyEnv_Def):
    def __init__(self):
        super().__init__(
            board_size=(10, 6),
            team_composition=[Soldier, Soldier, Archer, Archer, Knight, Knight],
            unit_types=[Soldier, Archer, Knight],
            max_turns=80,
            n_obstacles=6
        )

    def _setup_game(self):
        super()._setup_game()
        self._place_terrain()

    def _place_terrain(self):
        # Terreno vacío por defecto
        self.terrain = np.zeros(self.board_size, dtype=np.int8)

        forbidden = set()  # Evitar zonas de aparición y obstáculos
        for x in [0, 1, 8, 9]:  # Bandas iniciales de ambos equipos
            for y in range(self.board_size[1]):
                forbidden.add((x, y))
        forbidden.update([(x, y) for x in range(self.board_size[0]) for y in range(self.board_size[1]) if self.terrain[x, y] == 99])

        all_coords = [(x, y) for x in range(self.board_size[0]) for y in range(self.board_size[1]) if (x, y) not in forbidden]
        random.shuffle(all_coords)

        terrain_types = [1, 2, 3]  # bosque, colina, pantano
        for terrain_id in terrain_types:
            for _ in range(5):  # 5 casillas de cada tipo
                if all_coords:
                    x, y = all_coords.pop()
                    self.terrain[x, y] = terrain_id
