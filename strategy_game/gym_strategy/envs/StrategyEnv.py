import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gym_strategy.core.Board import Board
from gym_strategy.core.Unit import Soldier
from gym_strategy.core.Renderer import Renderer

class StrategyEnv(gym.Env):
    def __init__(self):
        super(StrategyEnv, self).__init__()
        self.board = Board(size=(6, 4))  # Tablero de 6 filas por 4 columnas
        self.units = []
        self.renderer = Renderer(width=600, height=400, board_size=(6, 4))
        self.current_turn = 0
        self.unit_index = 0

        # Tamaño del tablero
        self.rows, self.cols = 6, 4

        # Acción unificada: tipo (0 o 1) * fila * col
        self.action_space = spaces.Discrete(2 * self.rows * self.cols)

        # 🔢 Observación como matriz numérica: [equipo, salud]
        self.observation_space = spaces.Box(low=0, high=100, shape=(self.rows, self.cols, 2), dtype=np.float32)

        # 🖼️ Para usar observación visual en el futuro (captura de pantalla):
        # self.observation_space = spaces.Box(low=0, high=255, shape=(400, 600, 3), dtype=np.uint8)

        self.reset()

    def step(self, action):
        # Decodificamos acción única → tipo, coordenadas
        action_type, x, y = self.decode_action(action)

        current_team_units = [u for u in self.units if u.team == self.current_turn]
        if not current_team_units:
            return self._get_state(), -1, True, {}

        unit = current_team_units[self.unit_index]

        reward = -1  # Penalización base por acción inválida
        done = False

        if action_type == 0:
            reward = self.move_unit(unit, x, y)
            print(f"{unit.unit_type} intentó moverse a {(x, y)}.")
        elif action_type == 1:
            reward = self.attack_unit(unit, x, y)
            print(f"{unit.unit_type} intentó atacar a {(x, y)}.")

        # Cambio de unidad / turno
        self.unit_index += 1
        if self.unit_index >= len(current_team_units):
            self.unit_index = 0
            self.current_turn = 1 - self.current_turn
            print(f"🔄 Cambio de turno: Ahora juega el equipo {self.current_turn}.")

        self.render()
        done = self.check_game_over()
        return self._get_state(), reward, done, {}

    def move_unit(self, unit, new_x, new_y):
        if not (0 <= new_x < self.rows and 0 <= new_y < self.cols):
            print(f"❌ Movimiento fuera del tablero a ({new_x}, {new_y})")
            return -1

        if any(u.position == (new_x, new_y) for u in self.units):
            print(f"❌ Casilla ocupada en ({new_x}, {new_y})")
            return -1

        if self.board.is_valid_move((new_x, new_y)):
            unit.move((new_x, new_y))
            return 1  # Movimiento válido

        return -1

    def attack_unit(self, attacker, target_x, target_y):
        x, y = attacker.position

        if abs(target_x - x) + abs(target_y - y) != 1:
            return -1  # No es una casilla adyacente

        for unit in self.units:
            if unit.position == (target_x, target_y) and unit.team != attacker.team:
                unit.health -= 20
                if unit.health <= 0:
                    self.units.remove(unit)
                    return 5  # Elimina enemigo
                return 2  # Daña enemigo
        return -1  # Ataque inválido

    def check_game_over(self):
        team_0_units = [u for u in self.units if u.team == 0]
        team_1_units = [u for u in self.units if u.team == 1]
        return len(team_0_units) == 0 or len(team_1_units) == 0

    def reset(self):
        self.board = Board(size=(self.rows, self.cols))
        self.units = [
            Soldier((0, 0), team=0), Soldier((0, 1), team=0), Soldier((0, 2), team=0),
            Soldier((5, 1), team=1), Soldier((5, 2), team=1), Soldier((5, 3), team=1)
        ]
        for unit in self.units:
            self.board.add_unit(unit)
        self.current_turn = 0
        self.unit_index = 0
        return self._get_state()

    def render(self, mode="human"):
        if hasattr(self, "renderer"):
            self.renderer.draw_board(self.units)
        else:
            print(" Renderer no inicializado en StrategyEnv.")

    def _get_state(self):
        """
        Devuelve el estado como matriz numérica: canal 0 = equipo, canal 1 = salud.
        Si deseas cambiar a observación visual, este es el lugar donde hacerlo.
        """
        state = np.zeros((self.rows, self.cols, 2), dtype=np.float32)
        for unit in self.units:
            x, y = unit.position
            state[x, y, 0] = 1 if unit.team == 0 else -1
            state[x, y, 1] = unit.health
        return state

        # 🖼️ Alternativa para usar imagen capturada (para más adelante):
        # import pygame.surfarray
        # image = pygame.surfarray.array3d(self.renderer.screen)
        # image = np.transpose(image, (1, 0, 2))
        # return image

    # 🔢 Funciones helper para codificar/decodificar acciones
    def encode_action(self, action_type, x, y):
        """Convierte acción (tipo, x, y) en número entero."""
        return action_type * self.rows * self.cols + x * self.cols + y

    def decode_action(self, action):
        """Convierte acción entera en (tipo, x, y)."""
        action_type = action // (self.rows * self.cols)
        x = (action % (self.rows * self.cols)) // self.cols
        y = (action % (self.rows * self.cols)) % self.cols
        return action_type, x, y
