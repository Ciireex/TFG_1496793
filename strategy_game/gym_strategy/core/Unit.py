class Unit:
    def __init__(self, unit_type, position, team, movement_range, attack_range, attack_damage):
        self.unit_type = unit_type  # "Soldier", "Archer", "Knight"
        self.position = position  # (row, column)
        self.team = team  # 0 o 1
        self.health = 100
        self.movement_range = movement_range  # Rango de movimiento permitido
        self.attack_range = attack_range  # Rango de ataque permitido
        self.attack_damage = attack_damage  # Daño base

    def move(self, new_position, board_size):
        x, y = self.position
        new_x, new_y = new_position

        # Verifica si el movimiento está dentro del rango permitido
        if (new_x != x and new_y != y) or abs(new_x - x) > self.movement_range or abs(new_y - y) > self.movement_range:
            return False  # Movimiento inválido

        # Verifica que la nueva posición esté dentro del tablero
        if 0 <= new_x < board_size[0] and 0 <= new_y < board_size[1]:
            self.position = new_position
            return True
        return False  # Movimiento fuera del tablero

    def attack(self, other_unit):
        x, y = self.position
        ox, oy = other_unit.position

        # Verifica si el enemigo está en el rango de ataque permitido
        if abs(x - ox) + abs(y - oy) <= self.attack_range:
            other_unit.health -= self.attack_damage
            return True
        return False  # Ataque inválido

class Soldier(Unit):
    def __init__(self, position, team):
        super().__init__("Soldier", position, team, movement_range=2, attack_range=1, attack_damage=25)

class Archer(Unit):
    def __init__(self, position, team):
        super().__init__("Archer", position, team, movement_range=3, attack_range=3, attack_damage=15)

class Knight(Unit):
    def __init__(self, position, team):
        super().__init__("Knight", position, team, movement_range=4, attack_range=1, attack_damage=30)