import numpy as np
from sb3_contrib.ppo_mask import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from gym_strategy.envs.StrategyEnvCaptureMaskedDiscrete import StrategyEnvCaptureMaskedDiscrete
import time

def mask_fn(env):
    return env._get_action_mask()

def verificar_avanzado(model_path, partidas=1000):
    base_env = StrategyEnvCaptureMaskedDiscrete()
    env = ActionMasker(base_env, mask_fn)

    model = MaskablePPO.load(model_path)

    exitos = 0
    turnos_totales_exito = []
    turnos_totales_fallo = []

    tiempos_inicio = time.time()

    for partida in range(partidas):
        obs, info = env.reset()
        done = False
        turn = 0

        while not done:
            mask = info["action_mask"]
            action, _ = model.predict(obs, deterministic=True, action_masks=mask)
            obs, reward, done, truncated, info = env.step(action)
            turn += 1

        if reward >= 1.0:
            exitos += 1
            turnos_totales_exito.append(turn)
        else:
            turnos_totales_fallo.append(turn)

    tiempo_total = time.time() - tiempos_inicio

    porcentaje_exito = (exitos / partidas) * 100
    tiempo_medio_exito = np.mean(turnos_totales_exito) if turnos_totales_exito else 0
    tiempo_medio_fallo = np.mean(turnos_totales_fallo) if turnos_totales_fallo else 0

    print("\n📊 Resultados de la verificación:")
    print(f"✅ Capturas exitosas: {exitos}/{partidas} ({porcentaje_exito:.2f}%)")
    print(f"⏱️ Turnos medios (éxito): {tiempo_medio_exito:.2f}")
    print(f"⏱️ Turnos medios (fallo): {tiempo_medio_fallo:.2f}")
    print(f"⏳ Tiempo total verificación: {tiempo_total:.2f} segundos")

    # Guardar resultados en archivo de texto
    with open("verificacion_resultados.txt", "w") as f:
        f.write("Resultados de la verificación:\n")
        f.write(f"Capturas exitosas: {exitos}/{partidas} ({porcentaje_exito:.2f}%)\n")
        f.write(f"Turnos medios (éxito): {tiempo_medio_exito:.2f}\n")
        f.write(f"Turnos medios (fallo): {tiempo_medio_fallo:.2f}\n")
        f.write(f"Tiempo total verificación: {tiempo_total:.2f} segundos\n")

if __name__ == "__main__":
    verificar_avanzado("ppo_capture_masked_v11.zip", partidas=1000)
