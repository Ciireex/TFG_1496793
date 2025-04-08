import numpy as np

class HeuristicAgent:
    def __init__(self, team=0):
        self.team = team

    def get_action(self, obs):
        # Encuentra la unidad activa (canal 3 == 1)
        active_coords = np.argwhere(obs[:, :, 3] == 1)
        if len(active_coords) == 0:
            return [0, 0, 0]  # Acción nula por defecto

        x, y = active_coords[0]

        # Valor que representa al enemigo en el canal 0
        enemy_value = -1 if self.team == 0 else 1

        # Busca enemigos adyacentes
        for dx, dy, direction in [(-1, 0, 0), (1, 0, 1), (0, -1, 2), (0, 1, 3)]:
            tx, ty = x + dx, y + dy
            if 0 <= tx < obs.shape[0] and 0 <= ty < obs.shape[1]:
                if obs[tx, ty, 0] == enemy_value:
                    return [0, direction, 1]  # Ataca sin moverse

        # Si no hay enemigos cerca, intenta moverse hacia el más cercano
        enemy_coords = np.argwhere(obs[:, :, 0] == enemy_value)
        if len(enemy_coords) > 0:
            # Calcula la distancia Manhattan a cada enemigo
            dists = [abs(ex - x) + abs(ey - y) for ex, ey in enemy_coords]
            closest_enemy = enemy_coords[np.argmin(dists)]
            ex, ey = closest_enemy

            # Decide dirección del movimiento (simple y directo)
            if abs(ex - x) > abs(ey - y):
                direction = 1 if ex > x else 0
            else:
                direction = 3 if ey > y else 2

            return [1, direction, 0]  # Mueve 1 hacia el enemigo

        return [0, 0, 0]  # Acción nula por defecto
