import os, sys, gymnasium as gym, pygame, time, torch
from stable_baselines3 import PPO

# Añadir ruta base al proyecto
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from gym_strategy.envs.StrategyEnv_TransferSmall_1v1 import StrategyEnv_TransferSmall_1v1
from gym_strategy.utils.CustomCNN import CustomCNN
from gym_strategy.core.Renderer import Renderer

# Cargar modelos
model_blue = PPO.load("./logs/ppo_blue_vs_trained_red/final_model", custom_objects={"features_extractor_class": CustomCNN})
model_red = PPO.load("./logs/ppo_red_vs_random/final_model", custom_objects={"features_extractor_class": CustomCNN})

# Inicializar entorno y renderer
env = StrategyEnv_TransferSmall_1v1()
renderer = Renderer(board_size=env.board_size)

# Reset y forzar que empiece el equipo azul
obs, _ = env.reset()
env.current_player = 0
terminated = False

# Loop principal
while not terminated:
    renderer.draw_board(units=env.units, active_unit=env._get_active_unit())
    time.sleep(0.8)

    if env.current_player == 0:  # Azul
        action, _ = model_blue.predict(obs, deterministic=True)
    else:  # Rojo
        action, _ = model_red.predict(obs, deterministic=True)

    obs, _, terminated, _, _ = env.step(action)

pygame.quit()
