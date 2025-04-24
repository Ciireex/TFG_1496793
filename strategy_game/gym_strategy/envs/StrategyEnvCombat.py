# gym_strategy/envs/StrategyEnvCombat.py
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gym_strategy.core.Board    import Board
from gym_strategy.core.Unit     import Soldier, Archer     # Knight se usará más adelante
from gym_strategy.core.Renderer import Renderer
import random

class StrategyEnvCombat(gym.Env):
    """
    7×7 · 3 vs 3  (2 Soldier + 1 Archer)  ·  Sin captura
    Acción  MultiDiscrete([2, 7, 7, 2, 7, 7])    ← 6 componentes
        0  move_flag   (0 = no mover | 1 = mover)
        1  Δx_move codif 0‑6  →  Δx = v‑3  (−3…+3)
        2  Δy_move codif 0‑6
        3  act_type   (0 = Atacar | 1 = Pasar)
        4  Δx_attack  codif 0‑6
        5  Δy_attack  codif 0‑6
    """

    SIZE  = 7
    MAX_D = 3                   # Knight moverá 3 en fases posteriores

    # ───────────────────────────────────────────────────────────
    def __init__(self):
        super().__init__()
        self.rows = self.cols = self.SIZE

        self.board    = Board(size=(self.rows, self.cols))
        self.renderer = Renderer(width=700, height=700,
                                 board_size=(self.rows, self.cols))

        # Espacios Gym
        self.action_space = spaces.MultiDiscrete(
            [2, 2*self.MAX_D+1, 2*self.MAX_D+1, 2, 2*self.MAX_D+1, 2*self.MAX_D+1]
        )
        self.observation_space = spaces.Box(
            0.0, 1.0, shape=(self.rows, self.cols, 8), dtype=np.float32
        )

        # Estado
        self.units, self.turn_units = [], []
        self.current_turn = 0
        self.unit_index   = 0
        self.turn_count   = 0
        self.max_turns    = 70
        self.idle_cost    = -0.05          # coste por turno vivo

        self.reset()

    # ──────────────────────────── STEP ──────────────────────────
    def step(self, action):
        mv_f, mx_i, my_i, act_t, ax_i, ay_i = action
        dx,  dy  = mx_i - self.MAX_D, my_i - self.MAX_D      # movimiento
        adx, ady = ax_i - self.MAX_D, ay_i - self.MAX_D      # ataque

        # Unidad activa
        if not self.turn_units or self.unit_index >= len(self.turn_units):
            self.turn_units = [u for u in self.units if u.team == self.current_turn]
        if not self.turn_units:
            return self._get_obs(), -40, True, False, {}

        unit  = self.turn_units[self.unit_index]
        enemy = 1 - unit.team
        reward = self.idle_cost                          # coste » obliga a actuar

        # ── Movimiento opcional ──────────────────────────────────
        if mv_f:
            nx, ny = unit.position[0] + dx, unit.position[1] + dy
            if self._move(unit, nx, ny):
                dist = abs(dx) + abs(dy)
                reward += 1.0 + 0.5 * (dist / unit.movement)
            else:
                reward -= 1.0

        # ── Acción principal ─────────────────────────────────────
        if act_t == 0:                                   # ATACAR
            tx = unit.position[0] + adx
            ty = unit.position[1] + ady
            reward += self._attack(unit, tx, ty) + 0.5
        else:                                            # PASAR
            reward -= 3.0
            if self._count(enemy) > self._count(unit.team):
                reward -= 3.0

        # Bonus acercarse al rival
        d = self._closest_enemy_dist(unit)
        if d < 4: reward += (3 - d) * 0.3                # hasta +0.9

        # ── Avance de índices / turnos ──────────────────────────
        self.unit_index += 1
        if self.unit_index >= len(self.turn_units):
            self.unit_index   = 0
            self.current_turn = enemy
            self.turn_units   = random.sample(
                [u for u in self.units if u.team == self.current_turn],
                k=len([u for u in self.units if u.team == self.current_turn])
            )
            self.turn_count += 1

        self.render()

        # ── Fin de partida / truncado ────────────────────────────
        done = self._count(0) == 0 or self._count(1) == 0
        if done:
            reward += 100 if self._count(unit.team) > 0 else -40

        if self.turn_count >= self.max_turns:
            done = True
            reward -= 25

        return self._get_obs(), reward, done, False, {}

    # ────────────────────── AUXILIARES ─────────────────────────
    def _move(self, unit, x, y):
        if not (0 <= x < self.rows and 0 <= y < self.cols): return False
        if any(u.position == (x, y) for u in self.units):    return False
        if abs(unit.position[0]-x)+abs(unit.position[1]-y) > unit.movement: return False
        unit.move((x, y)); return True

    def _attack(self, unit, tx, ty):
        dx, dy = abs(tx-unit.position[0]), abs(ty-unit.position[1])
        dist   = dx + dy
        if unit.unit_type == "Archer":
            if not ((dx == 0 or dy == 0) and 2 <= dist <= 3):
                return -1.0
        else:                       # Soldier
            if dist != 1: return -1.0

        for tgt in list(self.units):
            if tgt.position == (tx, ty) and tgt.team != unit.team:
                dmg, base = self._attack_reward(unit.unit_type, tgt.unit_type)
                tgt.health -= dmg
                if tgt.health <= 0:
                    self.units.remove(tgt);  return base + 30   # kill
                return base + 10                                # hit
        return -1.0

    @staticmethod
    def _attack_reward(a, d):
        if a == "Soldier": return (45, 15) if d == "Archer" else (34, 10)
        if a == "Archer":  return (25, 12) if d == "Knight" else (15, 5)
        return (15, 5)

    def _closest_enemy_dist(self, unit):
        return min((abs(unit.position[0]-e.position[0])
                    + abs(unit.position[1]-e.position[1])
                    for e in self.units if e.team != unit.team), default=99)

    def _count(self, t): return sum(1 for u in self.units if u.team == t)

    # ───────────────── reset / obs / render ────────────────────
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        def spawn(team, top):
            rows  = range(self.rows//2) if top else range(self.rows-self.rows//2, self.rows)
            cells = [(x, y) for x in rows for y in range(self.cols)]
            p = random.sample(cells, 3)
            return [Soldier(p[0], team), Soldier(p[1], team), Archer(p[2], team)]

        self.board = Board(size=(self.rows, self.cols))
        self.units = spawn(0, True) + spawn(1, False)
        for u in self.units: self.board.add_unit(u)

        self.current_turn = 0
        self.unit_index   = 0
        self.turn_units   = random.sample([u for u in self.units if u.team == 0], 3)
        self.turn_count   = 0
        return self._get_obs(), {}

    def _get_obs(self):
        obs = np.zeros((self.rows, self.cols, 8), np.float32)
        active = self.turn_units[self.unit_index] if self.unit_index < len(self.turn_units) else None
        for u in self.units:
            x, y = u.position
            idx  = {"Soldier": 0, "Archer": 1, "Knight": 2}[u.unit_type]
            obs[x, y, idx if u.team == self.current_turn else idx+3] = 1.0
            obs[x, y, 6] = u.health / 100.0
            obs[x, y, 7] = 1.0 if u is active else 0.0
        return obs

    def render(self, mode="human"):
        import pygame; pygame.event.pump()
        self.renderer.draw_board(self.units)
