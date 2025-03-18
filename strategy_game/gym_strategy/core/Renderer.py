import pygame

class Renderer:
    def __init__(self, width=600, height=400, board_size=(6, 4)):
        pygame.init()
        self.board_size = board_size
        self.cell_width = width // board_size[0]
        self.cell_height = height // board_size[1]
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Strategy Game")
    
    def draw_board(self, units):
        self.screen.fill((255, 255, 255))  # Fondo blanco
        
        # Dibujar cuadrícula
        for x in range(self.board_size[0]):
            for y in range(self.board_size[1]):
                rect = pygame.Rect(x * self.cell_width, y * self.cell_height, self.cell_width, self.cell_height)
                pygame.draw.rect(self.screen, (0, 0, 0), rect, 1)
        
        # Dibujar unidades
        for unit in units:
            color = (0, 0, 255) if unit.team == 0 else (255, 0, 0)  # Azul para equipo 0, Rojo para equipo 1
            unit_rect = pygame.Rect(
                unit.position[0] * self.cell_width, 
                unit.position[1] * self.cell_height, 
                self.cell_width, 
                self.cell_height
            )
            pygame.draw.rect(self.screen, color, unit_rect)
        
        pygame.display.flip()