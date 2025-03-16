import pygame

class Renderer:
    def __init__(self, width=600, height=600):
        pygame.init()
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Strategy Game")
    
    def draw_board(self, grid):
        self.screen.fill((255, 255, 255))  # White background
        for x in range(10):
            for y in range(10):
                rect = pygame.Rect(x * 60, y * 60, 60, 60)
                pygame.draw.rect(self.screen, (0, 0, 0), rect, 1)
        pygame.display.flip()
