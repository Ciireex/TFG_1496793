import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gym_strategy.core.Board   import Board
from gym_strategy.core.Unit    import Soldier, Archer, Knight
from gym_strategy.core.Renderer import Renderer
import random

class StrategyEnvSimple(gym.Env):
    """
    ▶  Mapa 7 × 7, punto de captura en el centro (3,3)
    ▶  Movimiento multi‑casilla (Soldier/Archer = 2, Knight = 3)
    ▶  Acción compacta: MultiDiscrete([2, 7, 7, 3])   ⇒   294 combinaciones
        0  = move_flag   (0 = no mover | 1 = mover)
        1–2 = Δx, Δy     (codificados 0‑6 →  Δ = valor‑3  →  −3…+3)
        3  = act_type    (0 = Atacar | 1 = Capturar | 2 = Pasar)
    ▶  Observación 8‑canales one‑hot (aliados/enemigos por tipo + salud + activo)
    ▶  Reward shaping:
        •  Mover válido        +1.5  (+0.5 · dist/mv)
        •  Mover inválido      −2.0
        •  Pasar               −3.0  (−6 si rival progresa)
        •  Acercarse al punto  +0.4
        •  Cobertura aliada    +2 / +1.2 / +0.6  (dist 0/1/2)
        •  Castigo por alejarse mientras capturas  −1
        •  Incentivos Soldier→Archer, Archer rango óptimo, etc.
    ▶  Orden de unidades barajado cada turno para que todas participen
    """

    MAX_D = 3                      # Knight mueve 3 (rango máximo usado en acción)

    # ────────────────────────────────────────────────
    def __init__(self):
        super().__init__()
        self.rows = self.cols = 7
        self.board    = Board(size=(self.rows, self.cols))
        self.renderer = Renderer(width=700, height=700, board_size=(self.rows, self.cols))

        # Estado de partida
        self.units, self.turn_units = [], []
        self.current_turn = 0
        self.unit_index   = 0
        self.turn_count   = 0
        self.max_turns    = 80
        self.no_progress_turns = 0

        # Punto de captura
        self.capture_point   = (self.rows//2, self.cols//2)
        self.capture_progress= {0: 0, 1: 0}
        self.capture_max     = 3
        self.capturing_team  = None

        # Acción compacta 4‑dim
        self.action_space = spaces.MultiDiscrete([
            2,                       # move_flag
            2*self.MAX_D + 1,        # Δx  codificado 0‑6
            2*self.MAX_D + 1,        # Δy
            3                        # act_type
        ])

        # Observación 8 canales
        self.observation_space = spaces.Box(0.0, 1.0, (self.rows, self.cols, 8), np.float32)

        self.reset()

    # ────────────────────────────────────────────────
    def step(self, action):
        move_flag, dx_i, dy_i, act_type = action
        dx, dy = dx_i - self.MAX_D, dy_i - self.MAX_D   # Δ reales −3…+3

        # 1) Seleccionar unidad activa
        if not self.turn_units or self.unit_index >= len(self.turn_units):
            self.turn_units = [u for u in self.units if u.team == self.current_turn]
        if not self.turn_units:                             # equipo sin unidades
            return self._get_obs(), -40, True, False, {}

        unit   = self.turn_units[self.unit_index]
        enemy  = 1 - unit.team
        reward = -0.1
        orig   = unit.position

        # ─── FASE 1 · Mover ──────────────────────────
        if move_flag:
            mx, my = orig[0] + dx, orig[1] + dy
            mv_r = self._move_unit(unit, mx, my)
            if mv_r > 0:
                reward += 1.5                                     # mover válido
                dist_used = abs(dx) + abs(dy)
                reward += (dist_used / unit.movement) * 0.5
            else:
                reward -= 2.0                                     # mover inválido

        # ─── FASE 2 · Acción principal ───────────────
        if act_type == 0:                                         # ATACAR
            ax, ay = orig if not move_flag else unit.position
            atk_r  = self._try_attack(unit, ax+dx, ay+dy)         # usa mismos offsets
            reward += atk_r + 0.5
            # Soldier ataca Archer
            for tgt in self.units:
                if tgt.position == (ax+dx, ay+dy) and tgt.team == enemy and tgt.unit_type == "Archer":
                    reward += 2.0
            # bonus si atacas capturador
            if (ax+dx, ay+dy) == self.capture_point and self.capturing_team == enemy:
                reward += 5.0

        elif act_type == 1:                                       # CAPTURAR
            if unit.position == self.capture_point:
                if self.capturing_team in (None, unit.team):
                    self.capture_progress[unit.team] += 1
                    self.capturing_team = unit.team
                    reward += 10
                    if self.capture_progress[unit.team] >= self.capture_max:
                        return self._get_obs(), 100, True, False, {}
                else:
                    self.capture_progress = {0:0, 1:0}
                    self.capturing_team   = unit.team
                    reward += 3
            else:
                reward -= 3                                          # capturar fuera

        else:                                                       # PASAR
            reward -= 3.0
            if self.capture_progress[enemy] > 0:
                reward -= 3.0

        # ─── Incentivos por tipo / cobertura ────────
        after_d  = self.manhattan_distance(unit.position, self.capture_point)
        before_d = self.manhattan_distance(orig,          self.capture_point)
        if after_d < before_d:                            # se acercó al punto
            reward += 0.4

        # Archer rango óptimo
        if unit.unit_type == "Archer":
            d = self._closest_enemy_distance(unit)
            reward += 1.0 if 2 <= d <= 3 else (-1.0 if d < 2 else 0)

        # Cobertura aliada
        if self.capturing_team == unit.team and self.capture_progress[unit.team] > 0:
            if after_d == 0:   reward += 2.0
            elif after_d == 1: reward += 1.2
            elif after_d == 2: reward += 0.6
            elif after_d >= 4: reward -= 1.0

        # Diferencia de captura
        diff = self.capture_progress[unit.team] - self.capture_progress[enemy]
        reward += diff * 5.0

        # Penal si rival progresa
        if self.capture_progress[enemy] > 0:
            reward -= max(0, 3 - after_d) * 3.0

        # ─── Control de no‑progreso ──────────────────
        if diff == 0 and move_flag == 0 and act_type == 2:
            self.no_progress_turns += 1
        else:
            self.no_progress_turns = 0

        # ─── Avance de índice / turno ────────────────
        self.unit_index += 1
        if self.unit_index >= len(self.turn_units):
            self.unit_index = 0
            self.current_turn = enemy
            # barajar orden para que todas las fichas participen
            self.turn_units = random.sample(
                [u for u in self.units if u.team == self.current_turn],
                k=len([u for u in self.units if u.team == self.current_turn])
            )
            self.turn_count += 1

        self.render()

        # ─── Fin de partida / truncado ───────────────
        terminated = self._check_game_over()
        if terminated:
            winner = 1 - self.current_turn
            reward += 100 if unit.team == winner else -40

        truncated = False
        if self.turn_count >= self.max_turns:
            reward += 10; truncated = True
        if self.no_progress_turns >= min(40, self.max_turns // 2):
            reward -= 20; truncated = True

        return self._get_obs(), reward, terminated, truncated, {}

    # ────────────────────────────────────────────────
    #               FUNCIONES AUXILIARES
    # ────────────────────────────────────────────────
    def _move_unit(self, unit, x, y):
        if not (0 <= x < self.rows and 0 <= y < self.cols): return -0.5
        if any(u.position == (x, y) for u in self.units):   return -0.5
        if self.manhattan_distance(unit.position, (x, y)) > unit.movement:
            return -0.5
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
                    if tgt.position == self.capture_point:
                        self.capture_progress[tgt.team] = 0
                        self.capturing_team = None
                        rwd += 5
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

    def _closest_enemy_distance(self, unit):
        x1, y1 = unit.position
        ds = [abs(x1-x2)+abs(y1-y2) for (x2,y2) in
              (u.position for u in self.units if u.team != unit.team)]
        return min(ds) if ds else float("inf")

    @staticmethod
    def manhattan_distance(p1, p2):
        return abs(p1[0]-p2[0]) + abs(p1[1]-p2[1])

    def _check_game_over(self):
        return not any(u.team == 0 for u in self.units) or not any(u.team == 1 for u in self.units)

    # ────────────────────────────────────────────────
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.turn_count = self.no_progress_turns = 0
        self.capture_progress = {0:0, 1:0}
        self.capturing_team   = None

        def spawn(team, top):
            rows = range(self.rows//2) if top else range(self.rows - self.rows//2, self.rows)
            cells = [(x,y) for x in rows for y in range(self.cols)
                     if self.manhattan_distance((x,y), self.capture_point) >= 2]
            s = random.sample(cells, 3)
            return [Soldier(s[0], team), Soldier(s[1], team), Archer(s[2], team)]

        self.board = Board(size=(self.rows, self.cols))
        self.units = spawn(0, True) + spawn(1, False)
        for u in self.units: self.board.add_unit(u)

        self.current_turn = 0
        self.unit_index   = 0
        self.turn_units   = random.sample([u for u in self.units if u.team == 0], k=3)
        return self._get_obs(), {}

    # ────────────────────────────────────────────────
    def render(self, mode="human"):
        import pygame
        pygame.event.pump()
        self.renderer.draw_board(
            self.units,
            capture_point=self.capture_point,
            capture_progress=self.capture_progress,
            capture_max=self.capture_max,
            capturing_team=self.capturing_team
        )

    # ────────────────────────────────────────────────
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
