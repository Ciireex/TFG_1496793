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

        self.screen.fill((255, 255, 255))  # fondo blanco
        cell_width = self.width // self.board_size[0]
        cell_height = self.height // self.board_size[1]

        # Pintar celdas
        for x in range(self.board_size[0]):
            for y in range(self.board_size[1]):
                rect = pygame.Rect(x * cell_width, y * cell_height, cell_width, cell_height)

                # Obstáculos
                if blocked_positions and (x, y) in blocked_positions:
                    pygame.draw.rect(self.screen, (0, 0, 0), rect)
                    continue

                # Punto de captura
                if capture_point == (x, y):
                    pygame.draw.rect(self.screen, (255, 255, 150), rect)  # amarillo claro

                    if capture_progress is not None:
                        # Detectar si es dict o int
                        progress = capture_progress if isinstance(capture_progress, int) else capture_progress.get(capturing_team, 0)
                        filled = int((progress / capture_max) * cell_height)
                        bar_color = (0, 0, 255) if capturing_team == 0 else (255, 0, 0)
                        bar_rect = pygame.Rect(
                            x * cell_width + cell_width // 4,
                            (y + 1) * cell_height - filled,
                            cell_width // 2,
                            filled
                        )
                        pygame.draw.rect(self.screen, bar_color, bar_rect)
                else:
                    pygame.draw.rect(self.screen, (255, 255, 255), rect)

                pygame.draw.rect(self.screen, (0, 0, 0), rect, 1)  # bordes

        # Dibujar unidades
        for unit in units:
            ux, uy = unit.position
            x_pix = ux * cell_width
            y_pix = uy * cell_height
            color = (0, 0, 255) if unit.team == 0 else (255, 0, 0)
            margin = int(cell_width * 0.15)

            unit_rect = pygame.Rect(
                x_pix + margin,
                y_pix + margin,
                cell_width - 2 * margin,
                cell_height - 2 * margin
            )
            pygame.draw.rect(self.screen, color, unit_rect)

            # Barra de vida
            max_hp = 100
            hp_ratio = max(0, unit.health / max_hp)
            bar_width = cell_width - 2 * margin
            bar_height = 5
            bar_x = x_pix + margin
            bar_y = y_pix + 2
            bar_color = (0, 200, 0) if hp_ratio > 0.5 else (255, 200, 0) if hp_ratio > 0.25 else (200, 0, 0)
            pygame.draw.rect(self.screen, (100, 100, 100), (bar_x, bar_y, bar_width, bar_height))
            pygame.draw.rect(self.screen, bar_color, (bar_x, bar_y, int(bar_width * hp_ratio), bar_height))

        pygame.display.flip()

    def get_image(self):
        image = pygame.surfarray.array3d(self.screen)
        return np.transpose(image, (1, 0, 2))
