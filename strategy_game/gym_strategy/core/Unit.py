class Unit:
    def __init__(self, unit_type, position, team):
        self.unit_type = unit_type  # "Soldier", "Archer", "Knight"
        self.position = position  # (x, y)
        self.team = team  # 0 o 1
        self.health = 100
        self.movement = 2  # Por defecto

    def move(self, new_position):
        self.position = new_position

    def attack(self, other_unit):
        if other_unit:
            other_unit.health -= self.get_attack_damage()

    def get_attack_damage(self):
        return 20  # Daño por defecto

    def is_alive(self):
        return self.health > 0


class Soldier(Unit):
    def __init__(self, position, team):
        super().__init__("Soldier", position, team)
        self.movement = 2

    def get_attack_damage(self):
        return 25


class Archer(Unit):
    def __init__(self, position, team):
        super().__init__("Archer", position, team)
        self.movement = 3

    def get_attack_damage(self):
        return 15


class Knight(Unit):
    def __init__(self, position, team):
        super().__init__("Knight", position, team)
        self.movement = 4

    def get_attack_damage(self):
        return 30
