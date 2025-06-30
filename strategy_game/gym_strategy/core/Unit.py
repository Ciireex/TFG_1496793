class Unit:
    def __init__(self, unit_type, position, team):
        self.unit_type = unit_type  # "Soldier", "Archer", "Knight"
        self.position = position    # (x, y)
        self.team = team            # 0 o 1
        self.health = 100
        self.max_health = 100
        self.attack_range = 1       # Por defecto
        self.attack_type = "melee"  # "melee" o "ranged"

    def move(self, new_position):
        self.position = new_position

    def attack(self, other_unit, game_map=None):
        if other_unit:
            damage = self.get_attack_damage(other_unit)
            other_unit.health -= damage

    def get_attack_damage(self, other_unit):
        return 20 

    def is_alive(self):
        return self.health > 0
    
    def has_advantage_over(self, other_unit):
        triangle = {
            "Soldier": "Archer",
            "Archer": "Knight",
            "Knight": "Soldier"
        }
        return triangle.get(self.unit_type) == other_unit.unit_type

class Soldier(Unit):
    def __init__(self, position, team):
        super().__init__("Soldier", position, team)
        self.attack_range = 1
        self.attack_type = "melee"

    def get_attack_damage(self, other_unit):
        return 50 if other_unit.unit_type == "Archer" else 34


class Archer(Unit):
    def __init__(self, position, team):
        super().__init__("Archer", position, team)
        self.attack_range = 3
        self.attack_type = "ranged"

    def get_attack_damage(self, other_unit):
        return 50 if other_unit.unit_type == "Knight" else 25


class Knight(Unit):
    def __init__(self, position, team):
        super().__init__("Knight", position, team)
        self.attack_range = 1
        self.attack_type = "melee"

    def get_attack_damage(self, other_unit):
        return 50 if other_unit.unit_type == "Soldier" else 34

    def attack(self, other_unit, game_map=None):
        if other_unit:
            dx = other_unit.position[0] - self.position[0]
            dy = other_unit.position[1] - self.position[1]
            push_pos = (other_unit.position[0] + dx, other_unit.position[1] + dy)

            can_push = (
                game_map is not None and
                game_map.is_within_bounds(push_pos) and
                game_map.is_empty(push_pos)
            )

            if can_push:
                other_unit.position = push_pos
                damage = self.get_attack_damage(other_unit)
            else:
                damage = self.get_attack_damage(other_unit) + 10  # Más daño si no puede empujar

            other_unit.health -= damage