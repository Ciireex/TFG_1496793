import os
import sys
import pygame
import numpy as np
from stable_baselines3 import DQN, PPO, A2C
from sb3_contrib.ppo_mask import MaskablePPO

# === CONFIG ===
HUMAN_TEAM = 0
MODEL_PATH = "dqn_red_f7_v3.zip"

# === IMPORTS ===
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from gym_strategy.envs.StrategyEnv import Env_Fase7_Terreno
from gym_strategy.core.Renderer import Renderer
from gym_strategy.utils.CustomCNN_Pro2 import EnhancedTacticalFeatureExtractor

# === CARGAR MODELO ===
def load_model(path):
    if "maskableppo" in path.lower():
        return MaskablePPO.load(path, custom_objects={"features_extractor_class": EnhancedTacticalFeatureExtractor}, device="auto")
    elif "ppo" in path.lower():
        return PPO.load(path, custom_objects={"features_extractor_class": EnhancedTacticalFeatureExtractor}, device="auto")
    elif "a2c" in path.lower():
        return A2C.load(path, custom_objects={"features_extractor_class": EnhancedTacticalFeatureExtractor}, device="auto")
    elif "dqn" in path.lower():
        return DQN.load(path, device="auto")
    else:
        raise ValueError("Modelo no reconocido")

MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../models"))
model = load_model(os.path.join(MODEL_DIR, MODEL_PATH))

# === ENTORNO Y RENDERER ===
env = Env_Fase7_Terreno()
obs, _ = env.reset()
done = False

tile_size = 100
cols, rows = env.board_size[1], env.board_size[0]
width = int(tile_size * cols * 1.3)
height = int(tile_size * rows * 0.9)
renderer = Renderer(width=width, height=height, board_size=env.board_size)

# === SPRITES ===
SPRITE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../sprites"))
attack_frames = {}

def cargar_sprites():
    # Soldier Blue
    if os.path.exists(os.path.join(SPRITE_DIR, "SoldierBlue1.png")):
        sprite = pygame.image.load(os.path.join(SPRITE_DIR, "SoldierBlue1.png")).convert_alpha()
        renderer.override_sprite("Soldier_Team0", sprite)
        attack_frames[(0, "Soldier")] = [pygame.image.load(os.path.join(SPRITE_DIR, f"SoldierBlue{i}.png")).convert_alpha() for i in range(1, 7)]

    # Soldier Red
    if os.path.exists(os.path.join(SPRITE_DIR, "SoldierRed1.png")):
        sprite = pygame.image.load(os.path.join(SPRITE_DIR, "SoldierRed1.png")).convert_alpha()
        renderer.override_sprite("Soldier_Team1", sprite)
        attack_frames[(1, "Soldier")] = [pygame.image.load(os.path.join(SPRITE_DIR, f"SoldierRed{i}.png")).convert_alpha() for i in range(1, 7)]

    # Archer Blue
    if os.path.exists(os.path.join(SPRITE_DIR, "ArcherBlue1.png")):
        sprite = pygame.image.load(os.path.join(SPRITE_DIR, "ArcherBlue1.png")).convert_alpha()
        renderer.override_sprite("Archer_Team0", sprite)
        attack_frames[(0, "Archer")] = [pygame.image.load(os.path.join(SPRITE_DIR, f"ArcherBlue{i}.png")).convert_alpha() for i in range(1, 9)]

    # Archer Red
    if os.path.exists(os.path.join(SPRITE_DIR, "ArcherRed1.png")):
        sprite = pygame.image.load(os.path.join(SPRITE_DIR, "ArcherRed1.png")).convert_alpha()
        renderer.override_sprite("Archer_Team1", sprite)
        attack_frames[(1, "Archer")] = [pygame.image.load(os.path.join(SPRITE_DIR, f"ArcherRed{i}.png")).convert_alpha() for i in range(1, 9)]

    # Knight Blue
    if os.path.exists(os.path.join(SPRITE_DIR, "KnightBlue1.png")):
        sprite = pygame.image.load(os.path.join(SPRITE_DIR, "KnightBlue1.png")).convert_alpha()
        renderer.override_sprite("Knight_Team0", sprite)
        attack_frames[(0, "Knight")] = [pygame.image.load(os.path.join(SPRITE_DIR, f"KnightBlue{i}.png")).convert_alpha() for i in range(1, 7)]

    # Knight Red
    if os.path.exists(os.path.join(SPRITE_DIR, "KnightRed1.png")):
        sprite = pygame.image.load(os.path.join(SPRITE_DIR, "KnightRed1.png")).convert_alpha()
        renderer.override_sprite("Knight_Team1", sprite)
        attack_frames[(1, "Knight")] = [pygame.image.load(os.path.join(SPRITE_DIR, f"KnightRed{i}.png")).convert_alpha() for i in range(1, 9)]

cargar_sprites()

# === RENDER INICIAL ===
blocked = (env.terrain == 99).astype(np.int8)
renderer.draw_board(units=env.units, terrain=env.terrain,
                    blocked_positions=blocked, active_unit=env._get_active_unit())

clock = pygame.time.Clock()

# === INPUT HUMANO ===
print("Controles:")
print("W = ↑  S = ↓  A = ←  D = →")
print("Q = pasar turno")

KEY_TO_ACTION = {
    pygame.K_w: 3,
    pygame.K_s: 4,
    pygame.K_a: 1,
    pygame.K_d: 2,
    pygame.K_q: 0
}

def get_human_action():
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); exit()

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_F12:
                    pygame.image.save(renderer.screen, "captura.png")
                    print("🖼️ Pantallazo guardado como 'captura.png'")
                elif event.key in KEY_TO_ACTION:
                    return KEY_TO_ACTION[event.key]

# === LOOP PRINCIPAL ===
while not done:
    active_unit = env._get_active_unit()
    team = env.current_player
    action = None

    if team == HUMAN_TEAM:
        print(f"[HUMANO] Fase: {env.phase.upper()}  Unidad: {active_unit.unit_type} Pos: {active_unit.position}")
        action = get_human_action()
    else:
        action, _ = model.predict(obs, deterministic=True)
        action = int(action) if not isinstance(action, (np.ndarray, list)) else int(np.asarray(action).item())

    # Comprobación de ataque válido
    is_attack_phase = env.phase == "attack"
    is_attack_action = action in [1, 2, 3, 4]
    if is_attack_phase and is_attack_action:
        # Determinar la dirección y destino del ataque
        directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        dx, dy = directions[action - 1]
        tx, ty = active_unit.position[0] + dx, active_unit.position[1] + dy
        target = env._get_unit_at((tx, ty))

        print(f"[CHECK] Intento de ataque hacia ({dx}, {dy}) → Posición destino: ({tx}, {ty})")
        if target:
            print(f"[CHECK] Encontrado objetivo en ({tx}, {ty}): {target.unit_type}, team {target.team}")
        else:
            print(f"[CHECK] No hay objetivo en ({tx}, {ty})")

        key = (active_unit.team, active_unit.unit_type)
        frames = attack_frames.get(key)

        if active_unit.team == 0:  # Azul (mira a la derecha por defecto)
            flip = (dy == -1)  # Si ataca a la izquierda → invertir
        else:  # Rojo (mira a la izquierda por defecto)
            flip = (dy == 1)  # Si ataca a la derecha → invertir


        if frames:
            print(f"[ANIM] {active_unit.unit_type} del equipo {active_unit.team} ataca hacia ({dx}, {dy})")
            renderer.animate_attack(active_unit, frames, env.terrain, blocked_positions=blocked, target_position=(tx, ty), flip=flip)

    obs, _, done, _, _ = env.step(action)
    blocked = (env.terrain == 99).astype(np.int8)
    renderer.draw_board(units=env.units, terrain=env.terrain,
                        blocked_positions=blocked, active_unit=env._get_active_unit())
    clock.tick(3)
