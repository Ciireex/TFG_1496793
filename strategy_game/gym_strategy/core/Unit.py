class Unit:
    def __init__(self, unit_type, position, team):
        self.unit_type = unit_type  # "Soldier", "Archer", "Knight"
        self.position = position  # (row, column)
        self.team = team  # 0 or 1
        self.health = 100
        self.movement = 2  # Default movement range

    def move(self, new_position):
        self.position = new_position

    def attack(self, other_unit):
        if other_unit:
            other_unit.health -= self.get_attack_damage()

    def get_attack_damage(self):
        return 20  # Default attack damage

class Soldier(Unit):
    def __init__(self, position, team):
        super().__init__("Soldier", position, team)
        self.movement = 2  # Can move 2 tiles

    def get_attack_damage(self):
        return 25  # Soldiers have a slightly stronger attack

class Archer(Unit):
    def __init__(self, position, team):
        super().__init__("Archer", position, team)
        self.movement = 3  # Archers have more mobility

    def get_attack_damage(self):
        return 15  # Archers deal less damage but attack from range

class Knight(Unit):
    def __init__(self, position, team):
        super().__init__("Knight", position, team)
        self.movement = 4  # Knights have the highest mobility

    def get_attack_damage(self):
        return 30  # Knights deal heavy damage
