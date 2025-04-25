from sb3_contrib.ppo_mask import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from gym_strategy.envs.MiniMaskEnv import MiniMaskEnv
import numpy as np

def mask_fn(env):
    return env._get_obs()["action_mask"]

env = ActionMasker(MiniMaskEnv(), mask_fn)
model = MaskablePPO.load("ppo_minimask")

obs, _ = env.reset()

for i in range(5):
    print("🎯 Máscara:", obs["action_mask"])
    action, _ = model.predict(obs, deterministic=True)
    print(f"🤖 Acción elegida: {action}")
    assert obs["action_mask"][action] == 1, "❌ Acción inválida seleccionada"
    obs, reward, done, _, _ = env.step(action)
    if done:
        print(f"🏁 Recompensa: {reward}")
        obs, _ = env.reset()
