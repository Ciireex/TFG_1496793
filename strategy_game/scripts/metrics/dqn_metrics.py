import os
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# === Rutas DQN ===
BASE_LOG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../logs'))
BLUE_DIR = os.path.join(BASE_LOG_DIR, "dqn/blue_f7_v3/DQN_2")
RED_DIR = os.path.join(BASE_LOG_DIR, "dqn/red_f7_v3/DQN_1")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "../../graficas_tfevents")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# === Métricas clave para DQN ===
# DQN no tiene explained_variance ni policy loss → usamos solo estas
metricas_clave = [
    "rollout/ep_rew_mean",
    "rollout/ep_len_mean",
    "train/loss"  # En DQN es el total_loss
]

# === Cargar métricas ===
def cargar_metricas(path_dir, color):
    resultados = {tag: [] for tag in metricas_clave}
    files = [f for f in os.listdir(path_dir) if "tfevents" in f]

    for filename in files:
        path = os.path.join(path_dir, filename)
        print(f"📂 {color.capitalize()} - Procesando: {filename}")
        ea = event_accumulator.EventAccumulator(path)
        ea.Reload()

        for tag in metricas_clave:
            if tag in ea.Tags().get("scalars", []):
                events = ea.Scalars(tag)
                steps = [e.step for e in events]
                values = [e.value for e in events]
                resultados[tag].append((steps, values))

    return resultados

# === Cargar datos ===
blue_data = cargar_metricas(BLUE_DIR, "blue")
red_data = cargar_metricas(RED_DIR, "red")

# === Dibujar comparativa ===
fig, axs = plt.subplots(2, 2, figsize=(12, 8))
axs = axs.flatten()

for i, tag in enumerate(metricas_clave):
    ax = axs[i]
    for steps, values in blue_data[tag]:
        ax.plot(steps, values, color='blue', label='Blue' if i == 0 else "", alpha=0.8)
    for steps, values in red_data[tag]:
        ax.plot(steps, values, color='red', label='Red' if i == 0 else "", alpha=0.8)
    ax.set_title(tag)
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Value")
    ax.grid(True)
    if i == 0:
        ax.legend()

# Si hay menos de 4 métricas, desactiva el panel sobrante
if len(metricas_clave) < 4:
    for j in range(len(metricas_clave), 4):
        axs[j].axis("off")

plt.tight_layout()
output_path = os.path.join(OUTPUT_DIR, f"dqn_comparativa_blue_vs_red.png")
plt.savefig(output_path)
print(f"[✓] Guardado: {output_path}")
plt.show()
