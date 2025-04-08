import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gym_strategy.core.Board import Board
from gym_strategy.core.Unit import Soldier, Archer
from gym_strategy.core.Renderer import Renderer
import random

class StrategyEnvSA(gym.Env):
    def __init__(self):
        super().__init__()
        self.rows, self.cols = 6, 4
        self.board = Board(size=(self.rows, self.cols))
        self.renderer = Renderer(width=600, height=400, board_size=(self.rows, self.cols))

        self.units = []
        self.current_turn = 0
        self.unit_index = 0
        self.turn_units = []
        self.turn_count = 0
        self.max_turns = 50
        self.no_progress_turns = 0

        self.action_space = spaces.MultiDiscrete([5, 4, 2])
        self.observation_space = spaces.Box(
            low=np.array([[[-1.0, 0.0, 0.0, 0.0]] * self.cols] * self.rows, dtype=np.float32),
            high=np.array([[[1.0, 100.0, 3.0, 1.0]] * self.cols] * self.rows, dtype=np.float32),
            dtype=np.float32
        )

        self.reset()

    def step(self, action):
        move_dist, direction, sec_action = action

        if not self.turn_units or self.unit_index >= len(self.turn_units):
            self.turn_units = [u for u in self.units if u.team == self.current_turn]

        if not self.turn_units:
            winner = 1 - self.current_turn
            print(f"🏁 ¡El equipo {winner} ha ganado! (el equipo {self.current_turn} no tiene unidades)")
            return self._get_obs(), -40, True, False, {}

        unit = self.turn_units[self.unit_index]
        reward = -0.1
        print(f"🔸 {unit.unit_type} del equipo {unit.team} en {unit.position} → mueve {move_dist} hacia {direction}, acción {sec_action}")

        before = self.closest_enemy_distance(unit)
        reward += self.move_unit(unit, move_dist, direction)
        after = self.closest_enemy_distance(unit)

        if after < before:
            reward += 1.0

        if sec_action == 1:
            attack_result = self.try_attack(unit)
            reward += attack_result
            if attack_result <= 0:
                reward -= 5
                if before > 1:
                    reward -= 2
        else:
            if self.closest_enemy_distance(unit) > 1:
                reward += 0.1

        if reward > 1:
            self.no_progress_turns = 0
        else:
            self.no_progress_turns += 1

        self.unit_index += 1
        if self.unit_index >= len(self.turn_units):
            self.unit_index = 0
            self.current_turn = 1 - self.current_turn
            self.turn_units = [u for u in self.units if u.team == self.current_turn]
            self.turn_count += 1
            print(f"🔄 Cambio de turno: Ahora juega el equipo {self.current_turn}.")

        self.render()

        terminated = self.check_game_over()
        if terminated:
            winner = 1 - self.current_turn
            print(f"🏁 ¡El equipo {winner} ha ganado!")
            reward += 100 if unit.team == winner else -40

        truncated = self.turn_count >= self.max_turns or self.no_progress_turns >= 12
        if truncated:
            print("⏱️ Fin por turnos o falta de progreso. Nadie ha ganado.")
            reward -= 20

        return self._get_obs(), reward, terminated, truncated, {}

    def move_unit(self, unit, dist, direction):
        max_range = getattr(unit, "movement", 2)
        if dist == 0 or dist > max_range:
            return -0.5

        dx, dy = 0, 0
        if direction == 0: dx = -1
        elif direction == 1: dx = 1
        elif direction == 2: dy = -1
        elif direction == 3: dy = 1

        x, y = unit.position
        for _ in range(dist):
            new_x, new_y = x + dx, y + dy
            if not (0 <= new_x < self.rows and 0 <= new_y < self.cols):
                return -1
            if any(u.position == (new_x, new_y) for u in self.units):
                return -1
            x, y = new_x, new_y

        unit.move((x, y))
        return 0.2 * dist

    def try_attack(self, unit):
        x, y = unit.position

        if unit.unit_type == "Archer":
            min_range = 2
            max_range = 3
        else:
            min_range = 1
            max_range = 1

        best_target = None
        best_distance = float("inf")

        for target in self.units:
            if target.team == unit.team:
                continue

            tx, ty = target.position
            dist = abs(x - tx) + abs(y - ty)
            if min_range <= dist <= max_range and dist < best_distance:
                best_target = target
                best_distance = dist

        if best_target:
            if unit.unit_type == "Soldier":
                if best_target.unit_type == "Archer":
                    damage = 45
                    reward = 15
                else:
                    damage = 34
                    reward = 10
            elif unit.unit_type == "Archer":
                damage = 15
                reward = 5
            else:
                damage = unit.get_attack_damage()
                reward = 10

            best_target.health -= damage

            if best_target.health <= 0:
                self.units.remove(best_target)
                print(f"☠️ {unit.unit_type} eliminó a {best_target.unit_type} en {best_target.position}")
                return reward + 20

            print(f"🎯 {unit.unit_type} dañó a {best_target.unit_type} en {best_target.position}")
            return reward

        print(f"❌ {unit.unit_type} no encontró enemigos dentro de rango.")
        return -1

    def closest_enemy_distance(self, unit):
        x1, y1 = unit.position
        enemies = [u.position for u in self.units if u.team != unit.team]
        if not enemies:
            return float("inf")
        return min(abs(x1 - x2) + abs(y1 - y2) for (x2, y2) in enemies)

    def check_game_over(self):
        return not any(u.team == 0 for u in self.units) or not any(u.team == 1 for u in self.units)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.turn_count = 0
        self.no_progress_turns = 0

        def generate_team(team, top_half):
            positions = [
                (x, y) for x in (range(self.rows // 2) if top_half else range(self.rows // 2, self.rows))
                for y in range(self.cols)
            ]
            sample = random.sample(positions, 3)
            return [Soldier(sample[0], team), Soldier(sample[1], team), Archer(sample[2], team)]

        self.board = Board(size=(self.rows, self.cols))
        self.units = generate_team(0, True) + generate_team(1, False)
        for unit in self.units:
            self.board.add_unit(unit)

        self.current_turn = random.choice([0, 1])
        self.unit_index = 0
        self.turn_units = [u for u in self.units if u.team == self.current_turn]

        print(f"🔁 Reset — empieza el equipo {self.current_turn}")
        return self._get_obs(), {}

    def render(self, mode="human"):
        if hasattr(self, "renderer"):
            import pygame
            pygame.event.pump()
            self.renderer.draw_board(self.units)

    def _get_obs(self):
        obs = np.zeros((self.rows, self.cols, 4), dtype=np.float32)
        active_unit = None
        team_units = [u for u in self.units if u.team == self.current_turn]
        if self.unit_index < len(team_units):
            active_unit = team_units[self.unit_index]

        for unit in self.units:
            x, y = unit.position
            obs[x, y, 0] = 1 if unit.team == 0 else -1
            obs[x, y, 1] = unit.health
            obs[x, y, 2] = {"Soldier": 1, "Archer": 2, "Knight": 3}.get(unit.unit_type, 0)
            obs[x, y, 3] = 1 if unit == active_unit else 0

        return obs
