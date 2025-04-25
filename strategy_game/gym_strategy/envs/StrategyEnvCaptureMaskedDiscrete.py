import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
from gym_strategy.core.Unit import Soldier
from collections import deque

class StrategyEnvCaptureMaskedDiscrete(gym.Env):
    """
    Entorno 5×5 con obstáculos aleatorios (camino garantizado), una sola unidad (Soldier)
    que debe recorrer al punto de captura y permanecer 3 turnos para ganar.
    Usa action_space=Discrete(5) y devuelve 'action_mask' para MaskablePPO.
    Recompensa: +1.0 al completar la captura, -0.01 por cada paso sin capturar.
    """
    def __init__(self):
        super().__init__()
        self.board_size = (5, 5)
        self.max_turns = 50
        self.capture_turns_required = 3

        # 0: quieto, 1: arriba, 2: abajo, 3: izquierda, 4: derecha
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Dict({
            "obs": spaces.Box(0.0, 1.0, shape=(3, *self.board_size), dtype=np.float32),
            "action_mask": spaces.Box(0, 1, shape=(5,), dtype=np.int8),
        })

        self.reset()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.turn_count = 0
        self.capture_progress = 0

        # Generar obstáculos + spawn/punto garantizando camino
        all_cells = [(x,y) for x in range(5) for y in range(5)]
        while True:
            num_blocks = random.randint(5, 10)
            self.blocked_positions = set(random.sample(all_cells, num_blocks))
            free = [c for c in all_cells if c not in self.blocked_positions]
            if len(free) < 2:
                continue
            spawn, point = random.sample(free, 2)
            if self._has_path(spawn, point):
                self.spawn = spawn
                self.capture_point = point
                break

        self.units = [Soldier(position=self.spawn, team=0)]
        return self._get_obs(), {}

    def _get_obs(self):
        # capa0: obstáculos, capa1: unidad, capa2: punto captura
        board = np.zeros((3, *self.board_size), dtype=np.float32)
        for x,y in self.blocked_positions:
            board[0,x,y] = 1.0
        ux,uy = self.units[0].position
        board[1,ux,uy] = self.units[0].health / 100
        cx,cy = self.capture_point
        board[2,cx,cy] = 1.0

        # construir máscara de acciones válidas
        mask = np.zeros(5, dtype=np.int8)
        dirs = [(0,0),(0,-1),(0,1),(-1,0),(1,0)]
        for a,(dx,dy) in enumerate(dirs):
            nx,ny = ux+dx, uy+dy
            if 0 <= nx < 5 and 0 <= ny < 5 and (nx,ny) not in self.blocked_positions:
                mask[a] = 1

        return {"obs": board, "action_mask": mask}

    def step(self, action):
        self.turn_count += 1
        me = self.units[0]
        dirs = [(0,0),(0,-1),(0,1),(-1,0),(1,0)]
        dx,dy = dirs[int(action)]
        new_pos = (me.position[0]+dx, me.position[1]+dy)

        # mover si válido
        if self._valid_move(new_pos):
            me.move(new_pos)

        # recompensa
        if me.position == self.capture_point:
            self.capture_progress += 1
            if self.capture_progress >= self.capture_turns_required:
                reward = 1.0
                done = True
            else:
                reward = 0.0
                done = False
        else:
            reward = -0.01
            self.capture_progress = 0
            done = False

        # fin por turnos
        if self.turn_count >= self.max_turns and not done:
            done = True

        info = {}
        if done:
            info["episode"] = {"r": reward, "l": self.turn_count}

        return self._get_obs(), reward, done, False, info

    def _valid_move(self, pos):
        x,y = pos
        if not (0 <= x < 5 and 0 <= y < 5):
            return False
        if pos in self.blocked_positions:
            return False
        return True

    def _has_path(self, start, goal):
        from collections import deque
        visited = {start}
        queue = deque([start])
        while queue:
            x,y = queue.popleft()
            if (x,y) == goal:
                return True
            for dx,dy in [(0,-1),(0,1),(-1,0),(1,0)]:
                nx,ny = x+dx,y+dy
                if (0<=nx<5 and 0<=ny<5 
                    and (nx,ny) not in self.blocked_positions 
                    and (nx,ny) not in visited):
                    visited.add((nx,ny))
                    queue.append((nx,ny))
        return False
