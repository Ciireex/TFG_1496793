import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gym_strategy.core.Board import Board
from gym_strategy.core.Unit import Soldier, Knight
from gym_strategy.core.Renderer import Renderer
import random

class StrategyEnvFase1(gym.Env):
    def __init__(self):
        super().__init__()
        self.use_archers = False
        self.use_knights = False  # Fase 1: solo soldados
        self.use_capture = False

        self.rows = self.cols = 5
        self.action_space = spaces.MultiDiscrete([5, 3, 5])
        self.observation_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(self.rows, self.cols, 8),
            dtype=np.float32
        )

        self.board = Board(size=(self.rows, self.cols))
        self.renderer = Renderer(width=500, height=500, board_size=(self.rows, self.cols))

        self.step_penalty     = -0.05
        self.hit_reward       = +4.0
        self.kill_reward      = +25.0
        self.win_reward       = +100.0
        self.loss_penalty     = -100.0
        self.proximity_reward = +0.1
        self.approach_reward  = +0.2
        self.invalid_attack_penalty = -1.0

        self.reset()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.board = Board(size=(self.rows, self.cols))
        self.units = []

        positions_0 = random.sample([(x, y) for x in range(0, 2) for y in range(self.cols)], 2)
        positions_1 = random.sample([(x, y) for x in range(3, 5) for y in range(self.cols)], 2)

        for pos in positions_0:
            self.units.append(Soldier(pos, team=0))

        for pos in positions_1:
            self.units.append(Soldier(pos, team=1))

        for u in self.units:
            self.board.add_unit(u)

        self.current_turn = 0
        self.unit_index = 0
        self.turn_count = 0

        return self._get_obs(), {}

    def step(self, action):
        team_units = [u for u in self.units if u.team == self.current_turn]
        if not team_units:
            return self._get_obs(), self.loss_penalty, True, False, {}

        unit = team_units[self.unit_index]
        move_dir, act_mode, act_dir = action
        reward = self.step_penalty

        old_dist = self._closest_enemy_dist(unit)
        dx, dy = self._dir_to_delta(move_dir)
        self._move(unit, dx, dy)
        new_dist = self._closest_enemy_dist(unit)
        if new_dist < old_dist:
            reward += self.approach_reward

        adx, ady = self._dir_to_delta(act_dir)
        if act_mode == 1:  # atacar
            reward += self._attack(unit, adx, ady)

        elif act_mode == 2:  # capturar (si no permitido → castigo)
            if not self.use_capture or getattr(self, "capture_point", None) != unit.position:
                reward += -2.0

        elif act_mode == 0:  # pasar
            if self._closest_enemy_dist(unit) == 1:
                reward += -2.0  # castigo por no atacar si hay enemigo cerca

        if self._adjacent_ally(unit):
            reward += self.proximity_reward

        # Turno siguiente
        self.unit_index += 1
        if self.unit_index >= self._count(self.current_turn):
            self.unit_index = 0
            self.current_turn = 1 - self.current_turn
            self.turn_count += 1

        self.render()

        done = self._count(0) == 0 or self._count(1) == 0
        if done:
            winner = unit.team if self._count(unit.team) > 0 else 1 - unit.team
            reward += self.win_reward if unit.team == winner else self.loss_penalty

        return self._get_obs(), reward, done, False, {}
    
    def render(self, mode="human"):
        import pygame
        pygame.event.pump()
        self.renderer.draw_board(self.units)

    def _dir_to_delta(self, d):
        return {0: (0, 0), 1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}[d]

    def _move(self, unit, dx, dy):
        x, y = unit.position
        nx, ny = x + dx, y + dy
        if 0 <= nx < self.rows and 0 <= ny < self.cols and not self.board.is_occupied((nx, ny)):
            unit.move((nx, ny))
            return True
        return False

    def _attack(self, unit, dx, dy):
        x, y = unit.position
        tx, ty = x + dx, y + dy
        for tgt in list(self.units):
            if tgt.position == (tx, ty) and tgt.team != unit.team:
                tgt.health -= unit.get_attack_damage()
                if tgt.health <= 0:
                    self.units.remove(tgt)
                    return self.kill_reward
                return self.hit_reward
        return self.invalid_attack_penalty

    def _count(self, team):
        return sum(1 for u in self.units if u.team == team)

    def _adjacent_ally(self, unit):
        x0, y0 = unit.position
        return any(
            abs(x0 - u.position[0]) + abs(y0 - u.position[1]) == 1
            for u in self.units if u.team == unit.team and u is not unit
        )

    def _closest_enemy_dist(self, unit):
        x0, y0 = unit.position
        ds = [abs(x0 - u.position[0]) + abs(y0 - u.position[1])
              for u in self.units if u.team != unit.team]
        return min(ds) if ds else 10

    def _get_obs(self):
        obs = np.zeros((self.rows, self.cols, 8), dtype=np.float32)
        team_units = [u for u in self.units if u.team == self.current_turn]
        active = team_units[self.unit_index] if self.unit_index < len(team_units) else None

        for u in self.units:
            x, y = u.position
            base = 0 if u.team == self.current_turn else 3
            idx = {"Soldier": 0, "Archer": 1, "Knight": 2}[u.unit_type]
            obs[x, y, base + idx] = 1.0
            obs[x, y, 6] = u.health / 100.0
            obs[x, y, 7] = 1.0 if u is active else 0.0
        return obs
