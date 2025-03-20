import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gym_strategy.core.Board import Board
from gym_strategy.core.Unit import Soldier
from gym_strategy.core.Renderer import Renderer

class StrategyEnv(gym.Env):
    def __init__(self):
        super(StrategyEnv, self).__init__()
        self.board = Board(size=(6, 4))  
        self.units = [] 
        self.renderer = Renderer(width=600, height=400, board_size=(6, 4)) 
        self.current_turn = 0  
        self.unit_index = 0  
        
        self.action_type_space = spaces.Discrete(2)  
        self.position_space = spaces.MultiDiscrete([6, 4]) 
        
        self.observation_space = spaces.Box(low=0, high=100, shape=(6, 4, 2), dtype=np.float32)
        self.reset()  

    def step(self, action_type, position):
        x, y = position 

        # Obtener la unidad que le toca jugar
        current_team_units = [u for u in self.units if u.team == self.current_turn]

        if not current_team_units:
            return self._get_state(), -1, True, {}  # Si no hay unidades, termina el juego

        unit = current_team_units[self.unit_index]

        # Obtener acciones válidas para la unidad
        valid_moves, valid_attacks = self.get_valid_actions(unit)

        # Si la acción no es válida, forzar una acción correcta
        if action_type == 0 and (x, y) not in valid_moves:
            if valid_moves:  # Si hay movimientos válidos, elige el primero
                x, y = valid_moves[0]
                print(f"Acción inválida corregida: {unit.unit_type} en {unit.position} se mueve a {x, y}.")
            else:
                action_type = 1  # Forzar ataque si no puede moverse

        if action_type == 1 and (x, y) not in valid_attacks:
            if valid_attacks:  # Si hay ataques válidos, elige el primero
                x, y = valid_attacks[0]
                print(f"Acción inválida corregida: {unit.unit_type} en {unit.position} ataca a {x, y}.")
            else:
                print(f" {unit.unit_type} en {unit.position} no tiene acciones válidas. Pasa el turno.")
                self.unit_index += 1
                if self.unit_index >= len(current_team_units):
                    self.unit_index = 0
                    self.current_turn = 1 - self.current_turn 
                    print(f"Cambio de turno: Ahora juega el equipo {self.current_turn}.")
                return self._get_state(), -1, False, {} 

        reward = -1  
        done = False

        if action_type == 0: 
            reward = self.move_unit(unit, x, y)
            print(f" {unit.unit_type} en {unit.position} se ha movido a {(x, y)} correctamente.")
        elif action_type == 1:  # Ataque
            reward = self.attack_unit(unit, x, y)
            print(f" {unit.unit_type} en {unit.position} ha atacado a un enemigo en {(x, y)}.")

        # Avanzar al siguiente soldado del equipo actual
        self.unit_index += 1
        if self.unit_index >= len(current_team_units):
            self.unit_index = 0
            self.current_turn = 1 - self.current_turn  
            print(f"🔄 Cambio de turno: Ahora juega el equipo {self.current_turn}.")

        self.render() 
        done = self.check_game_over()
        return self._get_state(), reward, done, {}


    def move_unit(self, unit, new_x, new_y):
        x, y = unit.position

        # Verifica si el movimiento está dentro del rango permitido (0, 1 o 2 casillas en línea recta)
        if (new_x != x and new_y != y) or abs(new_x - x) > 2 or abs(new_y - y) > 2:
            return -1  

        # Verificar si la casilla de destino está ocupada
        if any(u.position == (new_x, new_y) for u in self.units):
            print(f" Movimiento inválido: {unit.unit_type} en {unit.position} intentó moverse a {(new_x, new_y)}, pero la casilla está ocupada.")
            return -1  

        if self.board.is_valid_move((new_x, new_y)):
            unit.move((new_x, new_y), self.board.size)
            return 1  

        return -1  

    def attack_unit(self, attacker, target_x, target_y):
        x, y = attacker.position
        
        # Solo puede atacar si está en una casilla adyacente (NO diagonal)
        if abs(target_x - x) + abs(target_y - y) != 1:
            return -1 

        # Busca la unidad enemiga en la posición de ataque
        for unit in self.units:
            if unit.position == (target_x, target_y) and unit.team != attacker.team:
                unit.health -= 20  
                if unit.health <= 0:
                    self.units.remove(unit)  
                    return 5  
                return 2  
        return -1  

    def check_game_over(self):
        team_0_units = [u for u in self.units if u.team == 0] 
        team_1_units = [u for u in self.units if u.team == 1]  
        return len(team_0_units) == 0 or len(team_1_units) == 0 

    def reset(self):
        self.board = Board(size=(6, 4))  
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
        state = np.zeros((6, 4, 2), dtype=np.float32) 
        for unit in self.units:
            x, y = unit.position
            state[x, y, 0] = 1 if unit.team == 0 else -1  
            state[x, y, 1] = unit.health  
        return state 
    
    def get_valid_actions(self, unit):
        valid_moves = []
        valid_attacks = []

        x, y = unit.position

        # Generar movimientos válidos
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                if (dx == 0 or dy == 0) and 0 <= x + dx < 6 and 0 <= y + dy < 4:
                    if self.board.is_valid_move((x + dx, y + dy)):
                        valid_moves.append((x + dx, y + dy))

        # Generar ataques válidos
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            target_x, target_y = x + dx, y + dy
            if 0 <= target_x < 6 and 0 <= target_y < 4:
                for enemy in self.units:
                    if enemy.position == (target_x, target_y) and enemy.team != unit.team:
                        valid_attacks.append((target_x, target_y))

        return valid_moves, valid_attacks
