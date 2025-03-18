import numpy as np

class Board:
    def __init__(self, size=(6, 4)):
        self.size = size
        self.grid = np.zeros(size)  # 0: vacío, 1: ocupado
        self.units = []

    def add_unit(self, unit):
        self.units.append(unit)
        self.grid[unit.position] = 1

    def move_unit(self, unit, new_pos):
        x, y = unit.position
        new_x, new_y = new_pos

        # Verifica si el movimiento está dentro del tablero
        if not (0 <= new_x < self.size[0] and 0 <= new_y < self.size[1]):
            return False  # Movimiento inválido

        # Verifica si la casilla de destino está libre
        if self.grid[new_x, new_y] == 0:
            self.grid[x, y] = 0  # Vacía la casilla anterior
            unit.move(new_pos)
            self.grid[new_x, new_y] = 1  # Marca la nueva casilla como ocupada
            return True
        return False  # Movimiento inválido (casilla ocupada)

    def is_valid_move(self, new_pos):
        x, y = new_pos
        return (0 <= x < self.size[0]) and (0 <= y < self.size[1]) and self.grid[x, y] == 0