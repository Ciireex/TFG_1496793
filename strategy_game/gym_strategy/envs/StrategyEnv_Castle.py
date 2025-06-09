import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
from gym_strategy.core.Unit import Soldier, Archer, Knight

class StrategyEnv_Castle(gym.Env):
    def __init__(self, use_obstacles=True, obstacle_count=10):
        super().__init__()
        self.board_size = (10, 7)
        self.castle_area = [(4, 2), (4, 3), (5, 2), (5, 3)]
        self.max_turns = 60
        self.castle_control = 0
        self.unit_types = [Soldier, Soldier, Archer, Knight] * 2
        self.num_units = 8
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(0, 1, shape=(20, *self.board_size), dtype=np.float32)
        self.use_obstacles = use_obstacles
        self.obstacle_count = obstacle_count
        self.reset()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.turn_count = 0
        self.current_player = 0
        self.unit_index_per_team = {0: 0, 1: 0}
        self.phase = "move"
        self.castle_control = 0
        self.units = []

        # Posiciones posibles: fila 0 para azul, fila 9 para rojo
        blue_positions = [(0, y) for y in range(1, 6)]
        red_positions = [(9, y) for y in range(1, 6)]

        # Aleatorizamos
        random.shuffle(blue_positions)
        random.shuffle(red_positions)

        # Crear unidades para azul
        for i, unit_class in enumerate(self.unit_types[:4]):
            pos = blue_positions[i]
            self.units.append(unit_class(position=pos, team=0))

        # Crear unidades para rojo
        for i, unit_class in enumerate(self.unit_types[4:]):
            pos = red_positions[i]
            self.units.append(unit_class(position=pos, team=1))

        # Generar obstáculos
        unit_positions = [u.position for u in self.units]
        self.obstacles = self._generate_obstacles(unit_positions, self.obstacle_count) if self.use_obstacles else np.zeros(self.board_size, dtype=np.int8)

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

        if self.phase == "move":
            dx, dy = dirs[action]
            new_pos = (unit.position[0] + dx, unit.position[1] + dy)
            if self._valid_move(new_pos):
                unit.move(new_pos)
            else:
                reward -= 0.3
            self.phase = "attack"
            return self._get_obs(), reward, False, False, {}

        # Fase de ataque
        dx, dy = dirs[action]
        attacked = False
        prev_control = self.castle_control

        for dist in range(1, 4 if unit.unit_type == "Archer" else 2):
            tx, ty = unit.position[0] + dx * dist, unit.position[1] + dy * dist
            if not self._valid_coord((tx, ty)):
                break

            # Atacar castillo
            if (tx, ty) in self.castle_area:
                if self.current_player == 0:
                    self.castle_control = min(5, self.castle_control + 1)
                else:
                    self.castle_control = max(-5, self.castle_control - 1)

                if self.castle_control != prev_control:
                    reward += 0.6
                attacked = True
                break

            # Atacar enemigo
            for enemy in self.units:
                if enemy.team != self.current_player and enemy.is_alive() and enemy.position == (tx, ty):
                    enemy.health -= unit.get_attack_damage(enemy)
                    reward += 0.2
                    attacked = True
                    break
            if attacked:
                break

        self._advance_phase()

        if self.turn_count >= self.max_turns:
            reward -= 3.0  # penalización por empate
            terminated = True

        if abs(self.castle_control) >= 5:
            self._terminated = True
            self._winner = 0 if self.castle_control >= 5 else 1
            terminated = True
            if self._winner == self.current_player:
                reward += 3.0
            else:
                reward -= 2.0

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
            and pos not in self.castle_area
            and not any(u.position == pos and u.is_alive() for u in self.units)
        )

    def _get_obs(self):
        obs = np.zeros((20, *self.board_size), dtype=np.float32)
        obs[0] = self.obstacles
        for (x, y) in self.castle_area:
            obs[1, x, y] = 1.0
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

        active_units = [u for u in self.units if u.team == self.current_player and u.is_alive()]
        if self.unit_index_per_team[self.current_player] < len(active_units):
            unit = active_units[self.unit_index_per_team[self.current_player]]
            x, y = unit.position
            obs[12, x, y] = 1.0
            if unit.unit_type == "Soldier": obs[13, x, y] = 1.0
            elif unit.unit_type == "Knight": obs[14, x, y] = 1.0
            elif unit.unit_type == "Archer": obs[15, x, y] = 1.0
            obs[16, x, y] = unit.health / 100.0

        obs[17, :, :] = 1.0 if self.phase == "attack" else 0.0
        obs[18, :, :] = float(self.current_player)
        obs[19, :, :] = self.castle_control / 5.0  # Normalizado -1 a 1
        return obs

    def _is_adjacent_block_too_long(self, obstacles, x, y):
        horiz = obstacles[x-1:x+2, y] if 0 < x < obstacles.shape[0] - 1 else np.array([0])
        vert = obstacles[x, y-1:y+2] if 0 < y < obstacles.shape[1] - 1 else np.array([0])
        return horiz.sum() >= 2 or vert.sum() >= 2

    def _generate_obstacles(self, units_positions, obstacle_count=10):
        attempts = 100
        mid_x = self.board_size[0] // 2

        # Ampliamos la zona prohibida: castillo y su alrededor + casillas centrales para que no se bloquee el paso
        prohibited = set()
        for (x, y) in self.castle_area:
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    nx, ny = x + dx, y + dy
                    if self._valid_coord((nx, ny)):
                        prohibited.add((nx, ny))

        # Zona central (como sugerencia): evita bloquear el centro del mapa (columnas 3–6 por ejemplo)
        for x in range(self.board_size[0]):
            for y in range(2, 5):
                prohibited.add((x, y))

        for _ in range(attempts):
            obstacles = np.zeros(self.board_size, dtype=np.int8)

            left_half_positions = [
                (x, y) for x in range(1, mid_x)
                for y in range(1, self.board_size[1] - 1)
                if (x, y) not in prohibited and (x, y) not in units_positions
            ]
            random.shuffle(left_half_positions)

            placed = 0
            for x, y in left_half_positions:
                mirror_x = self.board_size[0] - 1 - x
                mirror_pos = (mirror_x, y)

                if ((x, y) in prohibited or mirror_pos in prohibited or
                    self._is_adjacent_block_too_long(obstacles, x, y) or
                    self._is_adjacent_block_too_long(obstacles, mirror_x, y)):
                    continue

                if (x, y) in units_positions or mirror_pos in units_positions:
                    continue

                # Introduce aleatoriedad para evitar muros rectos
                if random.random() < 0.85:  # 85% de los intentos colocan el par simétrico
                    obstacles[x, y] = 1
                    obstacles[mirror_x, y] = 1
                    placed += 2
                else:  # 15% de los casos: solo se coloca en un lado (para romper simetría un poco)
                    obstacles[x, y] = 1
                    placed += 1

                if placed >= obstacle_count:
                    return obstacles

        raise Exception("No se pudo generar obstáculos equilibrados.")
