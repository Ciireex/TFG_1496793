from sb3_contrib.ppo_mask import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from gym_strategy.envs.MiniMaskEnv import MiniMaskEnv

def mask_fn(env):
    return env._get_obs()["action_mask"]

env = ActionMasker(MiniMaskEnv(), mask_fn)

model = MaskablePPO("MultiInputPolicy", env, verbose=1)
model.learn(total_timesteps=1000)
model.save("ppo_minimask")
