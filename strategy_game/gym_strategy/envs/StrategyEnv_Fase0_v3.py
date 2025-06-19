import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
from gym_strategy.core.Unit import Soldier, Archer, Knight

class StrategyEnv_Fase0_v3(gym.Env):
    def __init__(self):
        super().__init__()
        self.board_size = (6, 4)
        self.max_turns = 60
        self.unit_types = [Soldier, Archer, Knight]
        self.terrain_types = {
            0: "plain",
            1: "forest",
            2: "hill",
            3: "camp",
            4: "obstacle"
        }
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(20, *self.board_size),
            dtype=np.float32
        )
        self.reset()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.turn_count = 0
        self.current_player = 0
        self.unit_index_per_team = {0: 0, 1: 0}
        self.phase = "move"
        self.units = []
        self.consecutive_passive_turns = 0

        # Posiciones iniciales
        blue_positions = random.sample([(x, y) for x in range(2) for y in range(4)], 3)
        red_positions = random.sample([(x, y) for x in range(4, 6) for y in range(4)], 3)

        # Crear unidades
        self.units = [
            Soldier(position=blue_positions[0], team=0),
            Archer(position=blue_positions[1], team=0),
            Knight(position=blue_positions[2], team=0),
            Soldier(position=red_positions[0], team=1),
            Archer(position=red_positions[1], team=1),
            Knight(position=red_positions[2], team=1)
        ]

        # Generar terreno (solo obstáculos)
        self.terrain = np.zeros(self.board_size, dtype=np.int8)
        obstacle_positions = random.sample(
            [(x, y) for x in range(self.board_size[0]) 
            for y in range(self.board_size[1])
            if (x, y) not in blue_positions + red_positions],
            min(6, self.board_size[0] * self.board_size[1] - 6)
        )
        for x, y in obstacle_positions:
            self.terrain[x, y] = 4

        return self._get_obs(), {}

    def _get_obs(self):
        obs = np.zeros((20, *self.board_size), dtype=np.float32)
        
        # Capas 0-4: Terreno
        for x in range(self.board_size[0]):
            for y in range(self.board_size[1]):
                if self.terrain[x, y] > 0:
                    obs[self.terrain[x, y] - 1, x, y] = 1.0
        
        # Capas 5-10: Unidades aliadas/enemigas
        unit_layers = {
            (0, Soldier): 5, (0, Archer): 6, (0, Knight): 7,
            (1, Soldier): 8, (1, Archer): 9, (1, Knight): 10
        }
        for unit in self.units:
            if unit.is_alive():
                x, y = unit.position
                obs[unit_layers[(unit.team, type(unit))], x, y] = 1.0
                health_layer = 11 if unit.team == self.current_player else 12
                obs[health_layer, x, y] = unit.health / unit.max_health

        # Capas 13-15: Información táctica
        if active_unit := self._get_active_unit():
            x, y = active_unit.position
            obs[13, x, y] = 1.0  # Unidad activa
            
            # Movimientos válidos
            for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
                if self._valid_move((x + dx, y + dy)):
                    obs[14, x + dx, y + dy] = 1.0
            
            # Ataques posibles
            for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
                for dist in range(1, active_unit.attack_range + 1):
                    tx, ty = x + dx*dist, y + dy*dist
                    if self._valid_coord((tx, ty)):
                        obs[15, tx, ty] = 1.0

        # Capas 16-18: Estado del juego
        obs[16] = 1.0 if self.phase == "attack" else 0.0
        obs[17] = self.turn_count / self.max_turns
        obs[18] = self.current_player

        return obs

    def step(self, action):
        reward = 0
        moved = False
        attacked = False  # Inicializada al principio
        terminated = False
        info = {
            "actions": [],
            "damage_dealt": 0,
            "healing": 0,
            "strategic_moves": 0,
            "attack_performed": False,  # Añadido para consistencia
            "unit_killed": False,
            "game_won": False
        }

        active_unit = self._get_active_unit()
        if not active_unit:
            self._advance_turn()
            return self._get_obs(), reward, terminated, False, info

        # Fase de movimiento
        if self.phase == "move":
            dx, dy = [(0,0), (-1,0), (1,0), (0,-1), (0,1)][action]
            new_pos = (active_unit.position[0] + dx, active_unit.position[1] + dy)

            if action != 0 and self._valid_move(new_pos):
                active_unit.move(new_pos)
                moved = True  # Marcamos que hubo movimiento
                strategic_value = self._calculate_strategic_value(active_unit, new_pos)
                reward += strategic_value
                info["strategic_moves"] += strategic_value * 100

                # Recompensa por acercamiento táctico
                closest_enemy = min(
                    (u for u in self.units if u.team != self.current_player and u.is_alive()),
                    key=lambda u: abs(u.position[0] - new_pos[0]) + abs(u.position[1] - new_pos[1]),
                    default=None
                )
                if closest_enemy:
                    dist = abs(closest_enemy.position[0] - new_pos[0]) + abs(closest_enemy.position[1] - new_pos[1])
                    if dist <= active_unit.attack_range:
                        reward += 0.15 * (1 - (dist / active_unit.attack_range))
            else:
                reward -= 0.01

            self.phase = "attack"

        # Fase de ataque
        else:
            dx, dy = [(0,0), (-1,0), (1,0), (0,-1), (0,1)][action]
            
            for dist in range(1, active_unit.attack_range + 1):
                target_pos = (active_unit.position[0] + dx * dist, active_unit.position[1] + dy * dist)
                if not self._valid_coord(target_pos):
                    break

                for enemy in [u for u in self.units if u.team != self.current_player and u.is_alive()]:
                    if enemy.position == target_pos:
                        # Calcular daño con bonus de terreno
                        terrain_bonus = 1.3 if (self.terrain[active_unit.position] == 2 and isinstance(active_unit, Archer)) else 1.0
                        damage = int(active_unit.get_attack_damage(enemy) * terrain_bonus)
                        
                        enemy.health -= damage
                        info["damage_dealt"] += damage
                        attacked = True
                        info["attack_performed"] = True
                        
                        # Recompensa proporcional al daño
                        reward += 0.8 * (damage / enemy.max_health)
                        
                        if not enemy.is_alive():
                            info["unit_killed"] = True
                            # Recompensa por eliminar unidad
                            unit_rewards = {Archer: 2.5, Soldier: 3.0, Knight: 4.0}
                            reward += unit_rewards.get(type(enemy), 2.0)
                            
                            if self._team_won(self.current_player):
                                terminated = True
                                info["game_won"] = True
                                reward += 20.0
                        break

                if attacked:
                    break

            # Penalizaciones ajustadas
            if not attacked:
                reward -= 0.1 if self._enemies_in_range(active_unit) else 0.02

            self._advance_phase()

        # Curación en campamento
        if self.phase == "move" and self.unit_index_per_team[self.current_player] == 0:
            for unit in [u for u in self.units if u.team == self.current_player and u.is_alive()]:
                if self.terrain[unit.position] == 3:
                    heal_amount = min(5, unit.max_health - unit.health)
                    unit.health += heal_amount
                    reward += 0.1 * heal_amount
                    info["healing"] += heal_amount

        # Terminación por tiempo
        if self.turn_count >= self.max_turns:
            terminated = True
            ally_count = sum(1 for u in self.units if u.is_alive() and u.team == self.current_player)
            enemy_count = sum(1 for u in self.units if u.is_alive() and u.team != self.current_player)
            
            if ally_count > enemy_count:
                reward += ally_count * 0.3
            else:
                reward -= min(3.0, enemy_count * 0.2)

        # Control de pasividad
        self.consecutive_passive_turns = 0 if (attacked or moved) else self.consecutive_passive_turns + 1
        if self.consecutive_passive_turns > 3:
            reward -= 0.2 * self.consecutive_passive_turns

        return self._get_obs(), reward, terminated, False, info

    def _calculate_strategic_value(self, unit, new_pos):
        """Calcula el valor estratégico de una posición"""
        value = 0
        
        # Bonus por cobertura
        if self.terrain[new_pos] == 1:  # Bosque
            value += 0.08 if isinstance(unit, Soldier) else 0.03
            
        # Bonus por altura
        if self.terrain[new_pos] == 2:  # Colina
            value += 0.12 if isinstance(unit, Archer) else 0.05
            
        # Bonus por posición ofensiva
        for enemy in self.units:
            if enemy.team != unit.team and enemy.is_alive():
                dist = abs(enemy.position[0] - new_pos[0]) + abs(enemy.position[1] - new_pos[1])
                if isinstance(unit, Knight) and dist == 1:
                    value += 0.15
                elif dist <= unit.attack_range:
                    value += 0.1 * (1 - (dist / unit.attack_range))
        
        return min(value, 0.3)  # Limitar el bonus máximo

    def _enemies_in_range(self, unit):
        return any(
            u for u in self.units 
            if u.team != unit.team and u.is_alive()
            and (abs(u.position[0] - unit.position[0]) + abs(u.position[1] - unit.position[1])) <= unit.attack_range
        )

    def _get_active_unit(self):
        team_units = [u for u in self.units if u.team == self.current_player and u.is_alive()]
        return team_units[self.unit_index_per_team[self.current_player] % len(team_units)] if team_units else None

    def _valid_coord(self, pos):
        x, y = pos
        return 0 <= x < self.board_size[0] and 0 <= y < self.board_size[1]

    def _valid_move(self, pos):
        return (self._valid_coord(pos) 
                and not any(u.position == pos and u.is_alive() for u in self.units)
                and self.terrain[pos] != 4)

    def _advance_phase(self):
        if self.phase == "move":
            self.phase = "attack"
        else:
            self.phase = "move"
            self.unit_index_per_team[self.current_player] += 1
            if self.unit_index_per_team[self.current_player] >= len([u for u in self.units if u.team == self.current_player and u.is_alive()]):
                self._advance_turn()

    def _advance_turn(self):
        self.turn_count += 1
        self.current_player = 1 - self.current_player
        self.unit_index_per_team[self.current_player] = 0
        self.phase = "move"

    def _team_won(self, team):
        return all(not u.is_alive() for u in self.units if u.team != team)