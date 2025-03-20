class Unit:
    def __init__(self, unit_type, position, team, movement_range, attack_range, attack_damage):
        self.unit_type = unit_type  # "Soldier", "Archer", "Knight"
        self.position = position  
        self.team = team 
        self.health = 100
        self.movement_range = movement_range  
        self.attack_range = attack_range  
        self.attack_damage = attack_damage  

    def move(self, new_position, board_size):
        x, y = self.position
        new_x, new_y = new_position

        if (new_x != x and new_y != y) or abs(new_x - x) > self.movement_range or abs(new_y - y) > self.movement_range:
            return False  

        if 0 <= new_x < board_size[0] and 0 <= new_y < board_size[1]:
            self.position = new_position
            return True
        return False 

    def attack(self, other_unit):
        x, y = self.position
        ox, oy = other_unit.position

        if abs(x - ox) + abs(y - oy) <= self.attack_range:
            other_unit.health -= self.attack_damage
            return True
        return False  

class Soldier(Unit):
    def __init__(self, position, team):
        super().__init__("Soldier", position, team, movement_range=2, attack_range=1, attack_damage=25)

class Archer(Unit):
    def __init__(self, position, team):
        super().__init__("Archer", position, team, movement_range=3, attack_range=3, attack_damage=15)

class Knight(Unit):
    def __init__(self, position, team):
        super().__init__("Knight", position, team, movement_range=4, attack_range=1, attack_damage=30)