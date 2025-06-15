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
                   active_unit=None, highlight_attack=False,
                   terrain=None,
                   capture_score=None, max_capture=5,
                   castle_area=None, castle_hp=None):

        pygame.event.pump()
        self.screen.fill((255, 255, 255))
        cell_width = self.width // self.board_size[0]
        cell_height = self.height // self.board_size[1]

        for x in range(self.board_size[0]):
            for y in range(self.board_size[1]):
                rect = pygame.Rect(x * cell_width, y * cell_height, cell_width, cell_height)

                if terrain is not None and terrain[x, y] == 1:
                    pygame.draw.rect(self.screen, (144, 238, 144), rect)

                if blocked_positions is not None and blocked_positions[x, y] == 1:
                    pygame.draw.rect(self.screen, (100, 100, 100), rect)
                    pygame.draw.rect(self.screen, (0, 0, 0), rect, 1)
                    continue

                if castle_area and (x, y) in castle_area:
                    pygame.draw.rect(self.screen, (180, 180, 255), rect)

                pygame.draw.rect(self.screen, (0, 0, 0), rect, 1)

        for unit in units:
            if not unit.is_alive():
                continue

            ux, uy = unit.position
            x_pix = ux * cell_width
            y_pix = uy * cell_height
            color = (0, 0, 255) if unit.team == 0 else (255, 0, 0)
            margin = int(cell_width * 0.15)

            if unit.unit_type == "Soldier":
                unit_rect = pygame.Rect(x_pix + margin, y_pix + margin + 8,
                                        cell_width - 2 * margin, cell_height - 2 * margin - 8)
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
                points = [(cx, cy - size), (cx - size, cy + size), (cx + size, cy + size)]
                pygame.draw.polygon(self.screen, color, points)

            if (active_unit and unit.position == active_unit.position and unit.team == active_unit.team):
                pygame.draw.rect(self.screen, (255, 165, 0), (x_pix + 2, y_pix + 2, cell_width - 4, cell_height - 4), 3)

            max_hp = 100
            hp_ratio = max(0, unit.health / max_hp)
            bar_width = cell_width - 2 * margin
            bar_height = 6
            bar_x = x_pix + margin
            bar_y = y_pix + 4 + 8
            bar_color = (0, 200, 0) if hp_ratio > 0.5 else (255, 200, 0) if hp_ratio > 0.25 else (200, 0, 0)
            pygame.draw.rect(self.screen, (80, 80, 80), (bar_x, bar_y, bar_width, bar_height))
            pygame.draw.rect(self.screen, bar_color, (bar_x, bar_y, int(bar_width * hp_ratio), bar_height))

        if capture_score:
            font = pygame.font.SysFont(None, 26)
            score_text = font.render(f"Capturas AZUL: {capture_score[0]} / ROJO: {capture_score[1]}", True, (0, 0, 0))
            self.screen.blit(score_text, (10, self.height - 30))

        if castle_hp is not None:
            center_x = self.width // 2
            font = pygame.font.SysFont(None, 26)
            text = font.render(f"Castillo: {castle_hp}", True, (0, 0, 0))
            self.screen.blit(text, (center_x - text.get_width() // 2, self.height - 60))
            bar_width = 200
            bar_height = 12
            filled_portion = int((castle_hp + 5) / 10 * bar_width)
            pygame.draw.rect(self.screen, (150, 150, 150), (center_x - bar_width//2, self.height - 40, bar_width, bar_height))
            pygame.draw.rect(self.screen, (0, 0, 255) if castle_hp >= 0 else (200, 0, 0),
                             (center_x - bar_width//2, self.height - 40, filled_portion, bar_height))

        pygame.display.flip()

    def get_image(self):
        image = pygame.surfarray.array3d(self.screen)
        return np.transpose(image, (1, 0, 2))
