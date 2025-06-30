import pygame

class Renderer:
    def __init__(self, width=600, height=600, board_size=(10, 10)):
        pygame.init()
        self.width = width
        self.height = height
        self.board_size = board_size
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Strategy Game")
        self.sprites = {}
        self.units = []

    def override_sprite(self, unit_type, image_surface):
        if image_surface:
            self.sprites[unit_type] = image_surface

    def draw_board(self, units, blocked_positions=None, active_unit=None, terrain=None):
        pygame.event.pump()
        self.units = units
        self.screen.fill((255, 255, 255))
        cell_width = self.width // self.board_size[0]
        cell_height = self.height // self.board_size[1]

        for x in range(self.board_size[0]):
            for y in range(self.board_size[1]):
                rect = pygame.Rect(x * cell_width, y * cell_height, cell_width, cell_height)
                if terrain is not None:
                    terrain_type = terrain[x, y]
                    if terrain_type == 1:
                        pygame.draw.rect(self.screen, (34, 139, 34), rect)
                    elif terrain_type == 2:
                        pygame.draw.rect(self.screen, (205, 133, 63), rect)
                    elif terrain_type == 3:
                        pygame.draw.rect(self.screen, (144, 238, 144), rect)

                if blocked_positions is not None and blocked_positions[x, y] == 1:
                    pygame.draw.rect(self.screen, (100, 100, 100), rect)
                    pygame.draw.rect(self.screen, (0, 0, 0), rect, 1)
                    continue

                pygame.draw.rect(self.screen, (0, 0, 0), rect, 1)

        for unit in units:
            if not unit.is_alive():
                continue

            ux, uy = unit.position
            x_pix = ux * cell_width
            y_pix = uy * cell_height
            margin = int(cell_width * 0.15)
            color = (0, 0, 255) if unit.team == 0 else (255, 0, 0)

            sprite_key = f"{unit.unit_type}_Team{unit.team}"
            sprite = getattr(unit, "_temp_sprite", self.sprites.get(sprite_key))

            if sprite:
                sprite_w = int(cell_width * 2.0)
                sprite_h = int(cell_height * 2.0)
                offset_x = (cell_width - sprite_w) // 2
                offset_y = (cell_height - sprite_h) // 2 + 10

                flip = (unit.team == 1)
                sprite_scaled = pygame.transform.scale(sprite, (sprite_w, sprite_h))
                sprite_scaled = pygame.transform.flip(sprite_scaled, flip, False)

                self.screen.blit(sprite_scaled, (x_pix + offset_x, y_pix + offset_y))
            else:
                if unit.unit_type == "Soldier":
                    rect = pygame.Rect(x_pix + margin, y_pix + margin + 8,
                                       cell_width - 2 * margin, cell_height - 2 * margin - 8)
                    pygame.draw.rect(self.screen, color, rect)
                elif unit.unit_type == "Archer":
                    cx = x_pix + cell_width // 2
                    cy = y_pix + cell_height // 2 + 4
                    radius = min(cell_width, cell_height) // 3
                    pygame.draw.circle(self.screen, color, (cx, cy), radius)
                elif unit.unit_type == "Knight":
                    cx = x_pix + cell_width // 2
                    cy = y_pix + cell_height // 2
                    size = min(cell_width, cell_height) // 2 - 4
                    points = [(cx, cy - size), (cx - size, cy + size), (cx + size, cy + size)]
                    pygame.draw.polygon(self.screen, color, points)

            if active_unit and unit.position == active_unit.position and unit.team == active_unit.team:
                pygame.draw.rect(self.screen, (255, 0, 0),
                 (x_pix + 2, y_pix + 2, cell_width - 4, cell_height - 4), 3)


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

    def animate_attack(self, unit, frames, terrain, blocked_positions, target_position, flip=False):
        print(f"[RENDER] Iniciando animación de {unit.unit_type}_Team{unit.team} → {target_position}")
        for i, frame in enumerate(frames):
            print(f"[RENDER] Frame {i+1}/{len(frames)}")
            frame_to_use = pygame.transform.flip(frame, True, False) if flip else frame
            unit._temp_sprite = frame_to_use

            self.draw_board(units=self.units, terrain=terrain,
                            blocked_positions=blocked_positions, active_unit=unit)
            pygame.display.update()
            pygame.time.delay(80)

        if hasattr(unit, "_temp_sprite"):
            del unit._temp_sprite
        print("[RENDER] Animación terminada y sprite restaurado")
