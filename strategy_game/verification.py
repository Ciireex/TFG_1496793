from stable_baselines3 import PPO
from gym_strategy.envs.StrategyEnvPPOA2C2 import StrategyEnvPPOA2C2

# Cargar modelos
model_blue = PPO.load("models/ppo_vs_heuristic_v3")
model_red = PPO.load("models/ppo_rojo_vs_heuristica_azul.zip")

# Enfrentamientos
n_episodes = 100
blue_wins = 0
red_wins = 0
draws = 0

for i in range(n_episodes):
    env = StrategyEnvPPOA2C2()
    obs, _ = env.reset()
    done = False

    while not done:
        current_team = env.current_player
        model = model_blue if current_team == 0 else model_red
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, _ = env.step(action)

    # Evaluar resultado
    teams_alive = set(u.team for u in env.units if u.is_alive())
    if len(teams_alive) == 1:
        winner = list(teams_alive)[0]
        if winner == 0:
            blue_wins += 1
        else:
            red_wins += 1
        winner_text = 'Azul' if winner == 0 else 'Rojo'
    else:
        draws += 1
        winner_text = 'Empate'

    print(f"Partida {i+1}/{n_episodes} → Ganador: {winner_text}")

# Resultados finales
print("\n==== RESULTADOS ====")
print(f"🏆 Victorias Azul (PPO v3): {blue_wins}")
print(f"🏆 Victorias Rojo (PPO rojo): {red_wins}")
print(f"🤝 Empates: {draws}")
