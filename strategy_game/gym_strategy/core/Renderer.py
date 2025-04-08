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

    def draw_board(self, units):
        self.screen.fill((255, 255, 255))  # Fondo blanco

        cell_width = self.width // self.board_size[0]
        cell_height = self.height // self.board_size[1]

        # Dibujar líneas del tablero
        for x in range(self.board_size[0]):
            for y in range(self.board_size[1]):
                rect = pygame.Rect(x * cell_width, y * cell_height, cell_width, cell_height)
                pygame.draw.rect(self.screen, (0, 0, 0), rect, 1)

        # Dibujar unidades
        for unit in units:
            ux, uy = unit.position
            x_pix = ux * cell_width
            y_pix = uy * cell_height

            unit_rect = pygame.Rect(
                x_pix + 5,
                y_pix + 5,
                cell_width - 10,
                cell_height - 10
            )
            color = (0, 0, 255) if unit.team == 0 else (255, 0, 0)
            pygame.draw.rect(self.screen, color, unit_rect)

            # Dibujar barra de vida
            max_hp = 100
            hp_ratio = max(0, unit.health / max_hp)
            bar_width = cell_width - 10
            bar_height = 5
            bar_color = (0, 200, 0) if hp_ratio > 0.5 else (255, 200, 0) if hp_ratio > 0.25 else (200, 0, 0)

            pygame.draw.rect(self.screen, (100, 100, 100), (x_pix + 5, y_pix, bar_width, bar_height))  # fondo
            pygame.draw.rect(self.screen, bar_color, (x_pix + 5, y_pix, bar_width * hp_ratio, bar_height))  # vida

        pygame.display.flip()

    def get_image(self):
        """Captura el contenido de la pantalla y lo devuelve como imagen RGB [H, W, 3]."""
        image = pygame.surfarray.array3d(self.screen)
        return np.transpose(image, (1, 0, 2))  # [ancho, alto, canal] → [alto, ancho, canal]
