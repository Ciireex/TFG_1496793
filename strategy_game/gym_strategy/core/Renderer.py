import pygame
import numpy as np

class Renderer:
    def __init__(self, width=600, height=600, board_size=(10, 10)):
        pygame.init()
        self.width = width
        self.height = height
        self.board_size = board_size
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Strategy Game")

    def draw_board(self, units, blocked_positions=None,
                   capture_point=None, capture_progress=None, capture_max=3, capturing_team=None):

        pygame.event.pump()  # 🔥 Procesar eventos para evitar congelamiento

        self.screen.fill((255, 255, 255))  # fondo blanco
        cell_width = self.width // self.board_size[0]
        cell_height = self.height // self.board_size[1]

        # Pintar celdas
        for x in range(self.board_size[0]):
            for y in range(self.board_size[1]):
                rect = pygame.Rect(x * cell_width, y * cell_height, cell_width, cell_height)

                # Obstáculos
                if blocked_positions and (x, y) in blocked_positions:
                    pygame.draw.rect(self.screen, (100, 100, 100), rect)  # Gris oscuro
                    pygame.draw.rect(self.screen, (0, 0, 0), rect, 1)  # Borde negro fino
                    continue

                # Punto de captura
                if capture_point == (x, y):
                    pygame.draw.rect(self.screen, (255, 255, 100), rect)  # Amarillo fuerte

                    if capture_progress is not None:
                        for team in [0, 1]:
                            progress = capture_progress[team]
                            bar_width = (cell_width - 10) // 2
                            filled = int((progress / capture_max) * bar_width)
                            bar_height = 5
                            bar_color = (0, 0, 255) if team == 0 else (255, 0, 0)
                            x_offset = x * cell_width + 5 + (0 if team == 0 else bar_width + 2)
                            bar_rect = pygame.Rect(x_offset, y * cell_height + 4, filled, bar_height)
                            bg_rect = pygame.Rect(x_offset, y * cell_height + 4, bar_width, bar_height)
                            pygame.draw.rect(self.screen, (100, 100, 100), bg_rect)
                            pygame.draw.rect(self.screen, bar_color, bar_rect)
                else:
                    pygame.draw.rect(self.screen, (255, 255, 255), rect)

                # Bordes de todas las casillas
                pygame.draw.rect(self.screen, (0, 0, 0), rect, 1)

        # Dibujar unidades
        for unit in units:
            if not unit.is_alive():
                continue

            ux, uy = unit.position
            x_pix = ux * cell_width
            y_pix = uy * cell_height
            color = (0, 0, 255) if unit.team == 0 else (255, 0, 0)
            margin = int(cell_width * 0.15)

            if unit.unit_type == "Soldier":
                unit_rect = pygame.Rect(
                    x_pix + margin,
                    y_pix + margin + 8,
                    cell_width - 2 * margin,
                    cell_height - 2 * margin - 8
                )
                pygame.draw.rect(self.screen, color, unit_rect)

            elif unit.unit_type == "Archer":
                center_x = x_pix + cell_width // 2
                center_y = y_pix + cell_height // 2 + 4
                radius = min(cell_width, cell_height) // 3
                pygame.draw.circle(self.screen, color, (center_x, center_y), radius)

            elif unit.unit_type == "Knight":
                cx = x_pix + cell_width // 2
                cy = y_pix + cell_height // 2
                size = min(cell_width, cell_height) // 2 - 4
                points = [
                    (cx, cy - size),  # Arriba
                    (cx - size, cy + size),  # Abajo izquierda
                    (cx + size, cy + size)   # Abajo derecha
                ]
                pygame.draw.polygon(self.screen, color, points)

            max_hp = 100
            hp_ratio = max(0, unit.health / max_hp)
            bar_width = cell_width - 2 * margin
            bar_height = 6
            bar_x = x_pix + margin
            bar_y = y_pix + 4 + 8
            bar_color = (0, 200, 0) if hp_ratio > 0.5 else (255, 200, 0) if hp_ratio > 0.25 else (200, 0, 0)
            pygame.draw.rect(self.screen, (80, 80, 80), (bar_x, bar_y, bar_width, bar_height))
            pygame.draw.rect(self.screen, bar_color, (bar_x, bar_y, int(bar_width * hp_ratio), bar_height))

        pygame.display.flip()

    def get_image(self):
        image = pygame.surfarray.array3d(self.screen)
        return np.transpose(image, (1, 0, 2))  # [ancho, alto, canal] -> [alto, ancho, canal]
