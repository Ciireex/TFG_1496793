import pygame
import time
from gym_strategy.envs.StrategyEnv import StrategyEnv

def main():
    env = StrategyEnv()
    obs = env.reset()
    done = False

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        env.render()  

        action_type = env.action_type_space.sample()  
        position = env.position_space.sample()  

        obs, reward, done, _ = env.step(action_type, position)
        print(f"Acción: {action_type}, Posición: {position}, Recompensa: {reward}")

        time.sleep(0.5)  

    print("Juego terminado")

if __name__ == "__main__":
    main()
