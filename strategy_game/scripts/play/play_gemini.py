import os
import sys
import time
import pygame
from stable_baselines3 import PPO
import gymnasium as gym

# Asegúrate de que el path al módulo CustomCNN esté accesible
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# Importa tus clases necesarias
from gym_strategy.envs.StrategyEnv_Gemini import StrategyEnv_Gemini
from gym_strategy.core.Renderer import Renderer
from gym_strategy.utils.CustomCNN_Pro import CustomCNN  # Asegúrate de tener esta importación

class PPOvsPPOWrapper(gym.Wrapper):
    def __init__(self, base_env, model_team0, model_team1):
        super().__init__(base_env)
        self.model_team0 = model_team0
        self.model_team1 = model_team1
        self.current_player = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.current_player = self.env.current_player
        return obs, info

    def step(self, action):
        # Equipo 0 (PPO) actúa
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        if terminated or truncated:
            return obs, reward, terminated, truncated, info

        # Equipo 1 (PPO) actúa paso a paso
        while self.env.current_player != 0 and not (terminated or truncated):
            if self.env.current_player == 1:
                action, _ = self.model_team1.predict(obs)
            else:
                action = 0  # Por defecto para otros jugadores (si los hubiera)
            
            obs, _, terminated, truncated, info = self.env.step(action)

        return obs, reward, terminated, truncated, info

def play_ppo_vs_ppo_with_render(model_team0_path, model_team1_path, render_delay=400):
    # Cargar los modelos
    print(f"Cargando modelo para Equipo 0 (Azul) desde {model_team0_path}...")
    model_team0 = PPO.load(model_team0_path, custom_objects={"CustomCNN": CustomCNN}, device="cpu")
    
    print(f"Cargando modelo para Equipo 1 (Rojo) desde {model_team1_path}...")
    model_team1 = PPO.load(model_team1_path, custom_objects={"CustomCNN": CustomCNN}, device="cpu")
    
    # Crear el entorno y el wrapper
    base_env = StrategyEnv_Gemini(use_obstacles=True)
    env = PPOvsPPOWrapper(base_env, model_team0, model_team1)
    renderer = Renderer(width=700, height=600, board_size=env.unwrapped.board_size)
    
    # Inicializar el juego
    obs, _ = env.reset()
    done = False
    current_attack_dir = (0, 0)
    dirs = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]  # Direcciones correspondientes a las acciones

    def draw_all():
        team_units = [u for u in env.unwrapped.units if u.team == env.unwrapped.current_player and u.is_alive()]
        index = env.unwrapped.unit_index_per_team[env.unwrapped.current_player]
        active_unit = team_units[index] if index < len(team_units) else None

        # Resaltar celdas de ataque
        highlighted_cells = []
        if active_unit and env.unwrapped.phase == "attack":
            dx, dy = current_attack_dir
            max_range = 3 if active_unit.unit_type == "Archer" else 1
            for dist in range(1, max_range + 1):
                tx, ty = active_unit.position[0] + dx * dist, active_unit.position[1] + dy * dist
                if env.unwrapped._valid_coord((tx, ty)):
                    highlighted_cells.append((tx, ty))
                else:
                    break

        # Dibujar el tablero
        renderer.draw_board(
            units=env.unwrapped.units,
            blocked_positions=[(x, y) for x in range(env.unwrapped.board_size[0]) 
                             for y in range(env.unwrapped.board_size[1]) 
                             if env.unwrapped.obstacles[x, y] == 1],
            active_unit=active_unit,
            castle_area=env.unwrapped.castle_area,
            castle_hp=env.unwrapped.castle_control,
        )

        # Dibujar áreas de ataque resaltadas
        for x, y in highlighted_cells:
            cw = renderer.width // renderer.board_size[0]
            ch = renderer.height // renderer.board_size[1]
            rect = pygame.Rect(x * cw, y * ch, cw, ch)
            pygame.draw.rect(renderer.screen, (255, 100, 100, 150), rect, 3)

        pygame.display.flip()
        time.sleep(render_delay / 1000)

    # Bucle principal del juego
    while not done:
        draw_all()
        
        # Obtener acción del modelo actual
        if env.unwrapped.current_player == 0:
            action, _ = model_team0.predict(obs)
        else:
            action, _ = model_team1.predict(obs)
        
        # Actualizar dirección de ataque para visualización
        if env.unwrapped.phase == "attack" and action in range(1, 5):
            current_attack_dir = dirs[action]
        else:
            current_attack_dir = (0, 0)

        # Ejecutar la acción
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

    # Mostrar resultado final
    draw_all()
    time.sleep(2)  # Pausa para ver el resultado
    
    if env.unwrapped.castle_control > 0:
        print(f"¡El Equipo 0 (Azul) gana con {env.unwrapped.castle_control} puntos de control del castillo!")
    elif env.unwrapped.castle_control < 0:
        print(f"¡El Equipo 1 (Rojo) gana con {-env.unwrapped.castle_control} puntos de control del castillo!")
    else:
        print("¡El juego terminó en empate!")
    
    print(f"Turnos jugados: {env.unwrapped.turn_count}")
    pygame.quit()

if __name__ == "__main__":
    # Configuración de las rutas a los modelos
    MODEL_TEAM0_PATH = "self_play_models_custom_cnn/best_model_team0.zip"
    MODEL_TEAM1_PATH = "self_play_models_custom_cnn/best_model_team1.zip"
    
    # Ejecutar la partida con renderizado
    play_ppo_vs_ppo_with_render(
        model_team0_path=MODEL_TEAM0_PATH,
        model_team1_path=MODEL_TEAM1_PATH,
        render_delay=400  # Ajusta este valor para cambiar la velocidad
    )