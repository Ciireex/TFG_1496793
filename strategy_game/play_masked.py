import time
import numpy as np
from sb3_contrib.ppo_mask import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from gym_strategy.envs.StrategyEnvCaptureMaskedDiscrete import StrategyEnvCaptureMaskedDiscrete
from gym_strategy.core.Renderer import Renderer

DIRECTIONS = ["quieto", "arriba", "abajo", "izquierda", "derecha"]

def mask_fn(env):
    return env._get_obs()["action_mask"]

if __name__ == "__main__":
    # 1) Crear y envolver el entorno
    base_env = StrategyEnvCaptureMaskedDiscrete()
    env = ActionMasker(base_env, mask_fn)

    # 2) Cargar tu PPO maskeado
    model = MaskablePPO.load("ppo_capture_masked_v4")

    # 3) Renderer para visualizar
    renderer = Renderer(width=600, height=600, board_size=(5, 5))

    # 4) Reset
    obs, _ = env.reset()
    done = False

    while not done:
        mask = obs["action_mask"]
        valid = [DIRECTIONS[i] for i,v in enumerate(mask) if v]
        print(f"🎯 Acciones válidas: {valid}")

        # ← Aquí la única corrección: pasar `obs`, no `obs['obs']`
        action, _ = model.predict(
            obs,
            deterministic=True,
            action_masks=mask
        )
        action = int(action)
        print(f"🤖 Elegida: {DIRECTIONS[action]}")

        # 5) Step y render
        obs, reward, done, truncated, info = env.step(action)
        renderer.draw_board(
            units=base_env.units,
            capture_point=base_env.capture_point,
            capture_progress=base_env.capture_progress,
            capture_max=base_env.capture_turns_required,
            capturing_team=0
        )
        time.sleep(0.3)

    print("🏁 Recompensa final:", reward)
