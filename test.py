import time
from stable_baselines3 import PPO
from better_idea import StackBlocksEnv   # oder der entsprechende Modul-Name

def play_episodes(model_path: str, num_episodes: int = 20):
    # 1) Environment mit GUI erstellen
    env = StackBlocksEnv(gui=True, max_steps=5000, num_blocks=10)
    # 2) Modell laden
    model = PPO.load(model_path, env=env)

    for ep in range(1, num_episodes + 1):
        obs = env.reset()
        done = False
        total_reward = 0.0
        print(f"=== Episode {ep} ===")
        while not done:
            # Deterministische Aktion vom trainierten Policy-Netzwerk
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            total_reward += reward
            # Kurze Pause für flüssiges Rendering
            time.sleep(1/240.)
        print(f"Episode {ep} beendet mit Gesamt-Reward: {total_reward:.1f}")

    env.close()
 
if __name__ == "__main__":
    play_episodes("stacking_model_with_grasp.zip", num_episodes=20)