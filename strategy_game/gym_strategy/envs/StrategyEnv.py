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

    def reset2(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._setup_game()
        return self._get_obs(), {}

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.np_random, _ = gym.utils.seeding.np_random(seed)
        random.seed(int(self.np_random.integers(0, 2**32 - 1)))  # <-- conversión correcta
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

        # === Inicializar distancias para evitar farming por acercarse ===
        self.last_distances = {
            unit: 99 for unit in self.units if unit.team == self.current_player
        }

    def step(self, action):
        reward = 0
        terminated = False
        info = {}
        active_unit = self._get_active_unit()

        if not active_unit:
            self._advance_turn()
            return self._get_obs(), reward, terminated, False, info

        # ---------- FASE DE MOVIMIENTO ----------
        if self.phase == "move":
            dx, dy = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)][action]
            new_pos = (active_unit.position[0] + dx, active_unit.position[1] + dy)

            if action != 0 and self._valid_move(new_pos):
                active_unit.move(new_pos)
                reward += 0.05

                terrain_type = self.terrain[new_pos]
                if terrain_type == 1:            # Bosque
                    reward += 0.05
                elif terrain_type == 2:          # Colina
                    reward += 0.1
                # Campamento se gestiona tras avanzar de fase

                # --- Acercamiento a enemigo ---
                if active_unit in self.last_distances:
                    closest = min(
                        (u for u in self.units if u.team != self.current_player and u.is_alive()),
                        key=lambda e: abs(new_pos[0] - e.position[0]) + abs(new_pos[1] - e.position[1]),
                        default=None
                    )
                    if closest:
                        new_dist = abs(new_pos[0] - closest.position[0]) + abs(new_pos[1] - closest.position[1])
                        last_dist = self.last_distances[active_unit]
                        reward += 0.1 if new_dist < last_dist else -0.05 if new_dist > last_dist else 0
                        self.last_distances[active_unit] = new_dist
            else:
                reward -= 0.02

            self.phase = "attack"

        # ---------- FASE DE ATAQUE ----------
        else:
            attacked = False
            dx, dy = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)][action]

            # Alcance efectivo (+1 si arquero en colina)
            extra_range = 1 if isinstance(active_unit, Archer) and self.terrain[active_unit.position] == 2 else 0
            eff_range = active_unit.attack_range + extra_range

            for dist in range(1, eff_range + 1):
                tx, ty = active_unit.position[0] + dx * dist, active_unit.position[1] + dy * dist
                if not self._valid_coord((tx, ty)):
                    break
                target = next((u for u in self.units if u.team != self.current_player and u.is_alive() and u.position == (tx, ty)), None)
                if target:
                    prev_alive = target.is_alive()

                    # Bonificación por ventaja táctica
                    if active_unit.has_advantage_over(target):
                        reward += 0.2

                    # Ataque (incluye empuje del caballero si corresponde)
                    if isinstance(active_unit, Knight):
                        active_unit.attack(target, self)
                    else:
                        active_unit.attack(target)

                    # Bosque: reducción de daño neta (-5) restaurando vida
                    if self.terrain[target.position] == 1 and target.is_alive():
                        target.health = min(target.health + 5, target.max_health)

                    # Bonificación por disparo óptimo (Arquero)
                    if isinstance(active_unit, Archer):
                        manhattan = abs(tx - active_unit.position[0]) + abs(ty - active_unit.position[1])
                        reward += 0.3 if manhattan in [2, 3] else -0.1 if manhattan == 1 else 0

                    reward += 0.5  # recompensa base por atacar

                    if not target.is_alive() and prev_alive:
                        reward += 3.0
                        if self._team_won(self.current_player):
                            reward += 10.0
                            terminated = True

                    attacked = True
                    break

            if not attacked and self._enemies_in_range(active_unit):
                reward -= 0.2

            self._advance_phase()

        # ---------- EFECTO DE CAMPAMENTO (curación) ----------
        if self.phase == "move" and self.terrain[active_unit.position] == 3:
            if active_unit.health < active_unit.max_health:
                active_unit.health = min(active_unit.health + 10, active_unit.max_health)
                reward += 0.3

        # ---------- PENALIZACIÓN SI LA SIGUIENTE UNIDAD ESTÁ ENCERRADA ----------
        next_unit = self._get_active_unit()
        if next_unit and self.phase == "move":
            directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
            if not any(self._valid_move((next_unit.position[0] + dx, next_unit.position[1] + dy)) for dx, dy in directions):
                reward -= 0.2

        # === INCENTIVO UNIVERSAL PARA IR A POR EL ENEMIGO ===
        if active_unit and self.phase == "move":
            enemies = [u for u in self.units if u.team != self.current_player and u.is_alive()]
            if enemies:
                closest = self.get_closest_enemy(active_unit)
                if closest:
                    dist = self.manhattan_distance(active_unit.position, closest.position)
                    prev_dist = self.last_distances.get(active_unit, 99)

                    if dist < prev_dist:
                        reward += 0.2
                    elif dist > prev_dist:
                        reward -= 0.1
                    elif dist >= 3:
                        reward -= 0.1  # castigo por estar lejos y no avanzar

                    self.last_distances[active_unit] = dist

        # ---------- FIN POR TURNOS MÁXIMOS ----------
        if self.turn_count >= self.max_turns:
            terminated = True
            ally_alive = sum(u.is_alive() and u.team == self.current_player for u in self.units)
            enemy_alive = sum(u.is_alive() and u.team != self.current_player for u in self.units)
            reward -= 3.0
            if enemy_alive == 0:
                reward += 10.0
            elif ally_alive == 0:
                reward -= 10.0
            elif ally_alive > enemy_alive:
                reward += 1.0
            elif ally_alive < enemy_alive:
                reward -= 1.0
            else:
                reward -= 2.0

        return self._get_obs(), reward, terminated, False, info

    def get_closest_enemy(self, unit):
        enemies = [u for u in self.units if u.team != unit.team and u.is_alive()]
        if not enemies:
            return None
        return min(enemies, key=lambda e: self.manhattan_distance(unit.position, e.position))

    def manhattan_distance(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def is_counter(self, unit, target):
        if unit.unit_type == "Soldier" and target.unit_type == "Archer":
            return True
        if unit.unit_type == "Archer" and target.unit_type == "Knight":
            return True
        if unit.unit_type == "Knight" and target.unit_type == "Soldier":
            return True
        return False

    def move_unit(self, unit, direction):
        dxdy = [(0, -1), (1, 0), (0, 1), (-1, 0)]  # UP, RIGHT, DOWN, LEFT
        if direction >= 4:
            return False

        dx, dy = dxdy[direction]
        new_x = unit.position[0] + dx
        new_y = unit.position[1] + dy

        # Fuera de mapa
        if not (0 <= new_x < self.board_size[0] and 0 <= new_y < self.board_size[1]):
            return False

        # Ocupado por otra unidad
        if any(u.position == (new_x, new_y) and u.is_alive() for u in self.units):
            return False

        # Obstáculo
        if self.obstacle_map[new_x, new_y] == 1:
            return False

        unit.position = (new_x, new_y)
        return True

    def resolve_attack(self, unit, direction):
        dxdy = [(0, -1), (1, 0), (0, 1), (-1, 0)]  # UP, RIGHT, DOWN, LEFT
        if direction >= 4:
            return None, 0, None

        dx, dy = dxdy[direction]
        tx, ty = unit.position[0] + dx, unit.position[1] + dy

        if not (0 <= tx < self.board_size[0] and 0 <= ty < self.board_size[1]):
            return None, 0, None

        target = next((u for u in self.units if u.position == (tx, ty) and u.team != unit.team and u.is_alive()), None)
        if not target:
            return None, 0, None

        base_damage = 34
        if self.is_counter(unit, target):
            base_damage += 10  # bonus por triángulo

        damage = min(base_damage, target.health)
        target.health -= damage

        unit.last_attack_successful = True
        unit.last_kill = not target.is_alive()

        # CABALLERO: intentar empujar
        push_result = None
        if unit.unit_type == "Knight":
            px, py = tx + dx, ty + dy
            if not (0 <= px < self.board_size[0] and 0 <= py < self.board_size[1]):
                push_result = "obstacle"
            elif self.obstacle_map[px, py] == 1:
                push_result = "obstacle"
            elif any(u.position == (px, py) and u.is_alive() for u in self.units):
                push_result = "unit"
            else:
                target.position = (px, py)
                push_result = "moved"

        return target, damage, push_result

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

    def _get_obs(self):
        obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        unit_layer_offset = 1
        unit_map = {Soldier: 0, Archer: 1, Knight: 2}

        # Obstáculos
        for x in range(self.board_size[0]):
            for y in range(self.board_size[1]):
                if self.terrain[x, y] == 99:
                    obs[0, x, y] = 1.0

        # Unidades
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

        # Unidad activa y rangos
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

        # === Canales de terreno ===
        terrain_channels = {
            1: -6,  # bosque
            2: -5,  # colina
            3: -4   # pantano
        }
        for x in range(self.board_size[0]):
            for y in range(self.board_size[1]):
                terrain_type = self.terrain[x, y]
                if terrain_type in terrain_channels:
                    obs[terrain_channels[terrain_type], x, y] = 1.0

        # Info de fase, turno y jugador
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
    
    def get_action_mask(self):
        active_unit = self._get_active_unit()
        mask = np.zeros(self.action_space.n, dtype=bool)

        if not active_unit:
            mask[0] = True
            return mask

        if self.phase == "move":
            mask[0] = True  # No mover
            for i, (dx, dy) in enumerate([(-1, 0), (1, 0), (0, -1), (0, 1)], start=1):
                new_pos = (active_unit.position[0] + dx, active_unit.position[1] + dy)
                if self._valid_move(new_pos):
                    mask[i] = True
        else:  # "attack"
            mask[0] = True  # No atacar
            for i, (dx, dy) in enumerate([(-1, 0), (1, 0), (0, -1), (0, 1)], start=1):
                for dist in range(1, active_unit.attack_range + 1):
                    tx = active_unit.position[0] + dx * dist
                    ty = active_unit.position[1] + dy * dist
                    if not self._valid_coord((tx, ty)):
                        break
                    if any(u.position == (tx, ty) and u.team != active_unit.team and u.is_alive() for u in self.units):
                        mask[i] = True
                        break
        return mask
    
    def get_winner(self):
        alive_teams = set(u.team for u in self.units if u.is_alive())
        print(f"[DEBUG] Equipos vivos: {alive_teams}")  # 👈 Añade esto

        if len(alive_teams) == 1:
            winner = alive_teams.pop()
            print(f"[DEBUG] Gana el equipo {winner}")
            return winner
        elif len(alive_teams) == 0:
            return None  # Doble KO
        else:
            return None  # Ambos vivos

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

class Env_Fase6_Terreno(StrategyEnv_Def):
    def __init__(self):
        super().__init__(
            board_size=(6, 4),
            team_composition=[Soldier, Soldier, Archer, Archer, Knight, Knight],
            unit_types=[Soldier, Archer, Knight],
            max_turns=80,
            n_obstacles=2  # solo 2 obstáculos
        )

    def _setup_game(self):
        self.turn_count = 0
        self.current_player = 0
        self.unit_index_per_team = {0: 0, 1: 0}
        self.phase = "move"
        self.units = []

        # === Inicializar terreno vacío
        self.terrain = np.zeros(self.board_size, dtype=np.int8)

        # === Reservar zonas de aparición
        forbidden = set()
        for x in [0, 1, self.board_size[0] - 2, self.board_size[0] - 1]:
            for y in range(self.board_size[1]):
                forbidden.add((x, y))

        # === Colocar obstáculos
        all_coords = [(x, y) for x in range(self.board_size[0])
                      for y in range(self.board_size[1]) if (x, y) not in forbidden]
        random.shuffle(all_coords)
        for (x, y) in all_coords[:self.n_obstacles]:
            self.terrain[x, y] = 99  # obstáculo

        # === Colocar unidades y terrenos especiales
        self._place_units()
        self._place_terrain()

        self.last_distances = {
            unit: 99 for unit in self.units if unit.team == self.current_player
        }

    def _place_terrain(self):
        # Evitar casillas ya ocupadas por unidades u obstáculos
        forbidden = set()
        for x in [0, 1, self.board_size[0] - 2, self.board_size[0] - 1]:
            for y in range(self.board_size[1]):
                forbidden.add((x, y))
        forbidden.update([(x, y) for x in range(self.board_size[0])
                          for y in range(self.board_size[1]) if self.terrain[x, y] == 99])
        forbidden.update([u.position for u in self.units])  # evitar unidades

        all_coords = [(x, y) for x in range(self.board_size[0])
                      for y in range(self.board_size[1]) if (x, y) not in forbidden]
        random.shuffle(all_coords)

        terrain_types = [1, 2, 3]  # bosque, colina, pantano
        for terrain_id in terrain_types:
            if all_coords:
                x, y = all_coords.pop()
                self.terrain[x, y] = terrain_id

class Env_Fase7_Terreno(StrategyEnv_Def):
    def __init__(self):
        super().__init__(
            board_size=(8, 6),
            team_composition=[Soldier, Soldier, Archer, Archer, Knight, Knight],
            unit_types=[Soldier, Archer, Knight],
            max_turns=100,
            n_obstacles=4
        )

    def _setup_game(self):
        self.turn_count = 0
        self.current_player = 0
        self.unit_index_per_team = {0: 0, 1: 0}
        self.phase = "move"
        self.units = []

        # === Inicializar terreno vacío
        self.terrain = np.zeros(self.board_size, dtype=np.int8)

        # === Reservar zonas de aparición
        forbidden = set()
        for x in [0, 1, self.board_size[0]-2, self.board_size[0]-1]:
            for y in range(self.board_size[1]):
                forbidden.add((x, y))

        # === Colocar obstáculos sin cortar caminos
        self._place_obstacles_safe(forbidden)

        # === Colocar unidades
        self._place_units()

        # === Colocar 1 tipo de terreno especial (1 = bosque, 2 = colina, 3 = pantano)
        self._place_terrain_types(forbidden)

        self.last_distances = {
            unit: 99 for unit in self.units if unit.team == self.current_player
        }

    def _place_obstacles_safe(self, forbidden):
        from scipy.ndimage import label

        max_attempts = 100
        for _ in range(max_attempts):
            self.terrain[:, :] = 0
            all_coords = [(x, y) for x in range(self.board_size[0])
                          for y in range(self.board_size[1]) if (x, y) not in forbidden]
            random.shuffle(all_coords)
            for (x, y) in all_coords[:self.n_obstacles]:
                self.terrain[x, y] = 99

            # Comprobar conectividad con flood fill
            walkable = (self.terrain != 99).astype(np.int8)
            labeled, n = label(walkable)
            left_zone = labeled[0:2, :].max()
            right_zone = labeled[-2:, :].max()

            if left_zone != 0 and right_zone != 0 and left_zone == right_zone:
                return  # hay camino conectado

        raise ValueError("No se pudo colocar obstáculos sin cortar caminos")

    def _place_terrain_types(self, forbidden):
        terrain_types = [1, 2, 3]  # bosque, colina, pantano
        all_coords = [(x, y) for x in range(self.board_size[0])
                    for y in range(self.board_size[1])
                    if (x, y) not in forbidden and self.terrain[x, y] == 0]
        random.shuffle(all_coords)

        for terrain_id in terrain_types:
            if all_coords:
                x, y = all_coords.pop()
                self.terrain[x, y] = terrain_id

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.active_unit = self._get_active_unit()
        return self._get_obs(), {}
