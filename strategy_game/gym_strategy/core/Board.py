import numpy as np

class Board:
    def __init__(self, size=(10, 10)):
        self.size = size
        self.grid = np.zeros(size)  # 0: empty, 1: occupied
        self.units = []

    def add_unit(self, unit):
        self.units.append(unit)
        self.grid[unit.position] = 1

    def move_unit(self, unit, new_pos):
        if self.grid[new_pos] == 0:  # Only move if the tile is free
            self.grid[unit.position] = 0
            unit.move(new_pos)
            self.grid[new_pos] = 1
