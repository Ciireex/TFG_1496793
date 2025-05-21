import numpy as np

class HeuristicPolicy:
    def __init__(self, env):
        self.env = env

    def get_action(self, obs):
        team = self.env.current_player
        team_units = [u for u in self.env.units if u.team == team and u.is_alive()]
        idx = self.env.unit_index_per_team[team]

        if idx >= len(team_units):
            return 0  # Pasar turno por seguridad

        unit = team_units[idx]
        x, y = unit.position

        # Fase de movimiento
        if self.env.phase == "move":
            cx, cy = self.env.capture_point
            dx = np.sign(cx - x)
            dy = np.sign(cy - y)

            # Orden de preferencia: avanzar en x, luego y
            for move in [(dx, 0), (0, dy), (0, 0)]:
                mx, my = x + move[0], y + move[1]
                if self.env._valid_move((mx, my)):
                    if move == (0, 0):
                        return 0
                    elif move == (-1, 0):
                        return 1  # ←
                    elif move == (1, 0):
                        return 2  # →
                    elif move == (0, -1):
                        return 3  # ↑
                    elif move == (0, 1):
                        return 4  # ↓

            return 0  # Si no hay opción válida, pasar

        # Fase de ataque
        elif self.env.phase == "attack":
            dirs = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]
            for i, (dx, dy) in enumerate(dirs):
                for dist in range(1, 4 if unit.unit_type == "Archer" else 2):
                    tx, ty = x + dx * dist, y + dy * dist
                    if not self.env._valid_coord((tx, ty)):
                        break
                    for enemy in self.env.units:
                        if enemy.team != team and enemy.is_alive() and enemy.position == (tx, ty):
                            return i  # Atacar en esa dirección
            return 0  # No hay enemigos a rango, pasar

        return 0
