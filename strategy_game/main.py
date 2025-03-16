from gym_strategy.envs.StrategyEnv import StrategyEnv

env = StrategyEnv()
obs = env.reset()

done = False
while not done:
    action = env.action_space.sample()  # Random action
    obs, reward, done, _ = env.step(action)
    print(f"Reward: {reward}")
