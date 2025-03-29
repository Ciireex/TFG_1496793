import numpy as np

class Board:
    def __init__(self, size=(10, 10)):
        self.size = size
        self.grid = np.zeros(size)  # 0: vacío, 1: ocupado
        self.units = []

    def add_unit(self, unit):
        self.units.append(unit)
        self.grid[unit.position] = 1

    def move_unit(self, unit, new_pos):
        if self.grid[new_pos] == 0:  # Solo se mueve si la casilla está libre
            self.grid[unit.position] = 0
            unit.move(new_pos)
            self.grid[new_pos] = 1

    def is_occupied(self, pos):
        """Devuelve True si hay una unidad en la posición dada."""
        return any(u.position == pos for u in self.units)

    def is_valid_move(self, pos):
        x, y = pos
        return 0 <= x < self.size[0] and 0 <= y < self.size[1] and not self.is_occupied(pos)
