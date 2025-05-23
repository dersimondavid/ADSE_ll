import time
import numpy as np
import pybullet as p
import pybullet_data
import pygame
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from better_idea import StackBlocksEnv   # dein Env-Modul

# ---- 1) pygame helper ----
def init_pygame(width=640, height=480):
    pygame.init()
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("RL Training: Greifen & Stapeln")
    return screen, pygame.time.Clock()

def get_pybullet_frame(width, height):
    # Kamera so setzen, dass Tisch & Stapel gut sichtbar sind
    cam_target = [0.5, 0.0, 0.2]
    view = p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=cam_target,
        distance=1.0,
        yaw=45, pitch=-30, roll=0,
        upAxisIndex=2
    )
    proj = p.computeProjectionMatrixFOV(
        fov=60,
        aspect=float(width)/height,
        nearVal=0.1, farVal=5.0
    )
    _, _, rgb, _, _ = p.getCameraImage(
        width, height, view, proj,
        renderer=p.ER_BULLET_HARDWARE_OPENGL
    )
    img = np.reshape(rgb, (height, width, 4))[:,:,:3]
    # Drehen/Spiegeln für pygame
    frame = np.rot90(img, k=1)
    frame = np.flip(frame, axis=1)
    return frame

# ---- 2) Callback für Live-Rendering ----
class PygameRenderCallback(BaseCallback):
    def __init__(self, screen, clock, width, height, verbose=0):
        super().__init__(verbose)
        self.screen = screen
        self.clock = clock
        self.w = width
        self.h = height

    def _on_step(self) -> bool:
        # nur rendern, wenn wir eine echte Env haben
        if hasattr(self.training_env, "envs"):
            env = self.training_env.envs[0]
        else:
            env = self.training_env
        # Frame aus PyBullet holen
        frame = get_pybullet_frame(self.w, self.h)
        surf = pygame.surfarray.make_surface(frame)
        self.screen.blit(surf, (0,0))
        pygame.display.flip()
        # Event-Loop (damit Fenster reagiert)
        for evt in pygame.event.get():
            if evt.type == pygame.QUIT:
                env.close()
                pygame.quit()
                exit()
        # max 30 FPS
        self.clock.tick(30)
        return True

# ---- 3) Trainings-Skript ----
if __name__ == "__main__":
    # A) Lade dein vorheriges Modell
    model = PPO.load("stacking_model_with_grasp.zip")

    # B) Umgebung für 10 Blöcke im GUI-Modus (benötigt für getCameraImage)
    env = StackBlocksEnv(gui=True, max_steps=5000, num_blocks=10)

    # C) pygame initialisieren
    W, H = 640, 480
    screen, clock = init_pygame(W, H)

    # D) Callback instanziieren
    render_cb = PygameRenderCallback(screen, clock, W, H)

    # E) Modell auf neue Env setzen und weitertunen
    model.set_env(env)
    # Je nach Bedarf 500k–1M Steps zum Feintuning
    model.learn(total_timesteps=500_000, callback=render_cb)

    # F) Ergebnis speichern
    model.save("fine_tuned_10blocks_pygame.zip")

    # G) Fenster offen halten
    print("Training beendet. Drücke [x] im Fenster zum Schließen.")
    while True:
        for evt in pygame.event.get():
            if evt.type == pygame.QUIT:
                env.close()
                pygame.quit()
                exit()
        clock.tick(10)
