class Unit:
    def __init__(self, unit_type, position, team):
        self.unit_type = unit_type  # "Soldier", "Archer", "Knight"
        self.position = position  # (x, y)
        self.team = team  # 0 o 1
        self.health = 100
        self.movement = 2  # Por defecto
        self.attack_range = 1  # Por defecto
        self.attack_type = "melee"  # Puede ser "melee" o "ranged"

    def move(self, new_position):
        self.position = new_position

    def attack(self, other_unit):
        if other_unit:
            damage = self.get_attack_damage(other_unit)
            other_unit.health -= damage

    def get_attack_damage(self, other_unit):
        return 20  # Daño genérico por defecto

    def is_alive(self):
        return self.health > 0


class Soldier(Unit):
    def __init__(self, position, team):
        super().__init__("Soldier", position, team)
        self.movement = 2
        self.attack_range = 1
        self.attack_type = "melee"

    def get_attack_damage(self, other_unit):
        if other_unit.unit_type == "Archer":
            return 50  # Mata a un Archer en 2 golpes
        else:
            return 34


class Archer(Unit):
    def __init__(self, position, team):
        super().__init__("Archer", position, team)
        self.movement = 3
        self.attack_range = 3  # Ataca de 1 a 3 casillas
        self.attack_type = "ranged"

    def get_attack_damage(self, other_unit):
        if other_unit.unit_type == "Knight":
            return 50
        else:
            return 25


class Knight(Unit):
    def __init__(self, position, team):
        super().__init__("Knight", position, team)
        self.movement = 4
        self.attack_range = 1
        self.attack_type = "melee"

    def get_attack_damage(self, other_unit):
        if other_unit.unit_type == "Soldier":
            return 50
        else:
            return 34
