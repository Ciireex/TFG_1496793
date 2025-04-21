# gym_strategy/envs/StrategyEnvCombat.py
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gym_strategy.core.Board    import Board
from gym_strategy.core.Unit     import Soldier, Archer, Knight
from gym_strategy.core.Renderer import Renderer
import random

class StrategyEnvCombat(gym.Env):
    """
    Fase 1  ·  Solo combate ───────────────────────────────────────────────
      •  Mapa 7×7. 3 unidades vs 3 (2 Soldier + 1 Archer por equipo)
      •  NO existe todavía el punto de captura ni economía
      •  Acción compacta  MultiDiscrete([2, 7, 7, 2])
            0  move_flag (0 no mover | 1 mover)
            1  Δx  codif 0‑6 → Δ = val‑3  (−3…+3)
            2  Δy  igual
            3  act_type  (0 = Atacar | 1 = Pasar)
      •  Recompensa:
            +1.5  mover válido      (+0.5·dist/mv)
            –2    mover ilegal
            +30   eliminar enemigo
            +10   dañar
            –3/‑6 pasar (peor si enemigo tiene ventaja numérica)
            +100  victoria, –40 derrota
      •  Turno barajado: todas las fichas actúan.
    """

    MAX_D = 3             # Knight moverá 3 en fases posteriores
    MAP   = 7

    # ────────────────────────────────────────────────────────────────────
    def __init__(self):
        super().__init__()
        self.rows = self.cols = self.MAP

        self.board    = Board(size=(self.rows, self.cols))
        self.renderer = Renderer(width=700, height=700,
                                 board_size=(self.rows, self.cols))

        # Estado mutable
        self.units, self.turn_units = [], []
        self.current_turn = 0
        self.unit_index   = 0
        self.turn_count   = 0
        self.max_turns    = 60
        self.no_progress_turns = 0

        # ▶ Acción 4‑dim (sin Capturar)
        self.action_space = spaces.MultiDiscrete([
            2,                       # move_flag
            2*self.MAX_D + 1,        # Δx
            2*self.MAX_D + 1,        # Δy
            2                        # 0 Atacar | 1 Pasar
        ])

        # ▶ Observación 8 canales (igual que versión completa)
        self.observation_space = spaces.Box(
            0.0, 1.0, (self.rows, self.cols, 8), np.float32
        )

        self.reset()

    # ────────────────────────────────────────────────────────────────────
    def step(self, action):
        move_flag, dx_i, dy_i, act_type = action
        dx, dy = dx_i - self.MAX_D, dy_i - self.MAX_D

        # ── Seleccionar unidad activa ────────────────────────────────
        if not self.turn_units or self.unit_index >= len(self.turn_units):
            self.turn_units = [u for u in self.units if u.team == self.current_turn]
        if not self.turn_units:
            return self._get_obs(), -40, True, False, {}

        unit   = self.turn_units[self.unit_index]
        enemy  = 1 - unit.team
        reward = -0.1
        orig   = unit.position

        # ── Movimiento opcional ───────────────────────────────────────
        if move_flag:
            mx, my = orig[0] + dx, orig[1] + dy
            mv_r = self._move_unit(unit, mx, my)
            if mv_r > 0:
                dist_used = abs(dx)+abs(dy)
                reward += 1.5 + (dist_used/unit.movement)*0.5
            else:
                reward -= 2.0

        # ── Acción principal ─────────────────────────────────────────
        if act_type == 0:                                     # ATACAR
            ax, ay = unit.position
            reward += self._try_attack(unit, ax+dx, ay+dy) + 0.5

        else:                                                 # PASAR
            reward -= 3.0
            # penal extra si estamos en desventaja numérica
            if len([u for u in self.units if u.team == enemy]) \
               > len([u for u in self.units if u.team == unit.team]):
                reward -= 3.0

        # ── Penalización estancamiento ──────────────────────────────
        self.no_progress_turns = 0 if act_type == 0 else self.no_progress_turns + 1

        # ── Avanza índice / turno ───────────────────────────────────
        self.unit_index += 1
        if self.unit_index >= len(self.turn_units):
            self.unit_index = 0
            self.current_turn = enemy
            self.turn_units = random.sample(
                [u for u in self.units if u.team == self.current_turn],
                k=len([u for u in self.units if u.team == self.current_turn])
            )
            self.turn_count += 1

        self.render()

        # ── Terminar / Truncar ──────────────────────────────────────
        terminated = self._check_game_over()
        if terminated:
            winner = 1 - self.current_turn
            reward += 100 if unit.team == winner else -40

        truncated = False
        if self.turn_count >= self.max_turns:
            truncated, reward = True, reward - 20
        if self.no_progress_turns >= 30:
            truncated, reward = True, reward - 20

        return self._get_obs(), reward, terminated, truncated, {}

    # ───────────────────────── Auxiliares combat ─────────────────────────
    def _move_unit(self, unit, x, y):
        if not (0 <= x < self.rows and 0 <= y < self.cols): return -0.5
        if any(u.position == (x, y) for u in self.units):   return -0.5
        if abs(unit.position[0]-x)+abs(unit.position[1]-y) > unit.movement: return -0.5
        unit.move((x, y)); return 0.2

    def _try_attack(self, unit, tx, ty):
        dx, dy = abs(tx-unit.position[0]), abs(ty-unit.position[1])
        dist = dx + dy
        if unit.unit_type == "Archer":
            if not ((dx == 0 or dy == 0) and 2 <= dist <= 3): return -0.5
        else:
            if dist != 1: return -0.5

        for tgt in list(self.units):
            if tgt.position == (tx, ty) and tgt.team != unit.team:
                dmg, rwd = self._attack_reward(unit.unit_type, tgt.unit_type)
                tgt.health -= dmg
                if tgt.health <= 0:
                    self.units.remove(tgt)
                    return rwd + 30
                return rwd
        return -0.5

    @staticmethod
    def _attack_reward(a, d):
        if a == "Soldier":
            return (45, 15) if d == "Archer" else ((20, 5) if d == "Knight" else (34, 10))
        if a == "Archer":
            return (25, 12) if d == "Knight" else ((10, 4) if d == "Soldier" else (15, 5))
        if a == "Knight":
            return (40, 14) if d == "Soldier" else ((25, 6) if d == "Archer" else (30, 10))
        return (15, 5)

    def _check_game_over(self):
        return not any(u.team == 0 for u in self.units) or not any(u.team == 1 for u in self.units)

    # ─────────────────────────── Reset & Obs ─────────────────────────────
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.turn_count = self.no_progress_turns = 0

        def spawn(team, top):
            rows = range(self.rows//2) if top else range(self.rows - self.rows//2, self.rows)
            cells = [(x,y) for x in rows for y in range(self.cols)]
            s = random.sample(cells, 3)
            return [Soldier(s[0], team), Soldier(s[1], team), Archer(s[2], team)]

        self.board = Board(size=(self.rows, self.cols))
        self.units = spawn(0, True) + spawn(1, False)
        for u in self.units: self.board.add_unit(u)

        self.current_turn = 0
        self.unit_index   = 0
        self.turn_units   = random.sample([u for u in self.units if u.team == 0], k=3)
        return self._get_obs(), {}

    def _get_obs(self):
        obs = np.zeros((self.rows, self.cols, 8), np.float32)
        active = self.turn_units[self.unit_index] if self.unit_index < len(self.turn_units) else None
        for u in self.units:
            x, y = u.position
            idx = {"Soldier":0, "Archer":1, "Knight":2}[u.unit_type]
            if u.team == self.current_turn:
                obs[x, y, idx] = 1.0
            else:
                obs[x, y, idx+3] = 1.0
            obs[x, y, 6] = u.health / 100.0
            obs[x, y, 7] = 1.0 if u is active else 0.0
        return obs

    # ─────────────────────────── Render ──────────────────────────────────
    def render(self, mode="human"):
        import pygame
        pygame.event.pump()
        self.renderer.draw_board(self.units)
