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
        self.max_turns = 80  # 🔁 Fin anticipado si se exceden
        self.turn_count = 0

        self.rows, self.cols = 6, 4
        self.action_space = spaces.Discrete(2 * self.rows * self.cols)

        self.observation_space = spaces.Box(
            low=np.array([[-1.0, 0.0]] * self.rows * self.cols).reshape(self.rows, self.cols, 2),
            high=np.array([[1.0, 100.0]] * self.rows * self.cols).reshape(self.rows, self.cols, 2),
            dtype=np.float32
        )

        self.reset()

    def step(self, action):
        action_type, x, y = self.decode_action(action)

        current_team_units = [u for u in self.units if u.team == self.current_turn]
        if not current_team_units:
            winner = 1 - self.current_turn
            print(f"🏁 ¡El equipo {winner} ha ganado! (el equipo {self.current_turn} no tiene unidades)")
            obs = self._get_state()
            reward = -40
            terminated = True
            truncated = False
            info = {}
            return obs, reward, terminated, truncated, info

        unit = current_team_units[self.unit_index]
        reward = -0.1  # Penalización base por turno gastado

        # 🧭 Ejecutar acción
        if action_type == 0:
            reward += self.move_unit(unit, x, y)
            print(f"{unit.unit_type} intentó moverse a {(x, y)}.")
        elif action_type == 1:
            reward += self.attack_unit(unit, x, y)
            print(f"{unit.unit_type} intentó atacar a {(x, y)}.")

        # 🔁 Cambio de unidad / turno
        self.unit_index += 1
        if self.unit_index >= len(current_team_units):
            self.unit_index = 0
            self.current_turn = 1 - self.current_turn
            self.turn_count += 1
            print(f"🔄 Cambio de turno: Ahora juega el equipo {self.current_turn}.")

        self.render()

        # 🏁 Comprobaciones de fin de partida
        terminated = self.check_game_over()
        if terminated:
            winner = 1 - self.current_turn
            print(f"🏁 ¡El equipo {winner} ha ganado!")
            if unit.team == winner:
                reward += 100  # ✅ Victoria
            else:
                reward -= 40  # ❌ Derrota

        truncated = self.turn_count >= self.max_turns
        if truncated:
            print("⏱️ Fin por turnos. Nadie ha ganado.")
            reward -= 20  # ❌ Empate

        info = {}
        return self._get_state(), reward, terminated, truncated, info

    def move_unit(self, unit, new_x, new_y):
        if not (0 <= new_x < self.rows and 0 <= new_y < self.cols):
            print(f"❌ Movimiento fuera del tablero a ({new_x}, {new_y})")
            return -2

        if any(u.position == (new_x, new_y) for u in self.units):
            print(f"❌ Casilla ocupada en ({new_x}, {new_y})")
            return -2

        old_x, old_y = unit.position
        distance = abs(new_x - old_x) + abs(new_y - old_y)
        if distance > 2:
            print(f"❌ Movimiento demasiado largo ({distance} > 2)")
            return -2

        if self.board.is_valid_move((new_x, new_y)):
            unit.move((new_x, new_y))
            return 0.2  # ✅ Movimiento válido

        return -2

    def attack_unit(self, attacker, target_x, target_y):
        x, y = attacker.position

        if abs(target_x - x) + abs(target_y - y) != 1:
            return -2  # No adyacente ortogonal

        for unit in self.units:
            if unit.position == (target_x, target_y) and unit.team != attacker.team:
                unit.health -= 34  # 💥 Daño aumentado para morir en 3 golpes
                if unit.health <= 0:
                    self.units.remove(unit)
                    print(f"💥 {attacker.unit_type} eliminó a un enemigo en {(target_x, target_y)}")
                    return 30
                print(f"🎯 {attacker.unit_type} dañó a un enemigo en {(target_x, target_y)}")
                return 10

        return -2  # Ataque inválido

    def check_game_over(self):
        team_0_units = [u for u in self.units if u.team == 0]
        team_1_units = [u for u in self.units if u.team == 1]
        return len(team_0_units) == 0 or len(team_1_units) == 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.turn_count = 0

        self.board = Board(size=(self.rows, self.cols))
        self.units = [
            Soldier((0, 0), team=0), Soldier((0, 1), team=0), Soldier((0, 2), team=0),
            Soldier((5, 1), team=1), Soldier((5, 2), team=1), Soldier((5, 3), team=1)
        ]
        for unit in self.units:
            self.board.add_unit(unit)

        self.current_turn = 0
        self.unit_index = 0
        return self._get_state(), {}

    def render(self, mode="human"):
        if hasattr(self, "renderer"):
            self.renderer.draw_board(self.units)
        else:
            print(" Renderer no inicializado en StrategyEnv.")

    def _get_state(self):
        state = np.zeros((self.rows, self.cols, 2), dtype=np.float32)
        for unit in self.units:
            x, y = unit.position
            state[x, y, 0] = 1 if unit.team == 0 else -1
            state[x, y, 1] = unit.health
        return state

    def encode_action(self, action_type, x, y):
        return action_type * self.rows * self.cols + x * self.cols + y

    def decode_action(self, action):
        action_type = action // (self.rows * self.cols)
        x = (action % (self.rows * self.cols)) // self.cols
        y = (action % (self.rows * self.cols)) % self.cols
        return action_type, x, y
