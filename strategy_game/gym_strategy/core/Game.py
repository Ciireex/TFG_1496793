from gym_strategy.core import Board

class Game:
    def __init__(self):
        self.board = Board()
        self.players = [[], []]  # Units per team

    def add_unit(self, unit):
        self.board.add_unit(unit)
        self.players[unit.team].append(unit)

    def player_turn(self, player):
        print(f"Player {player}'s turn")

    def check_victory(self):
        if not self.players[0]:
            return 1  # Player 1 wins
        if not self.players[1]:
            return 0  # Player 0 wins
        return None
