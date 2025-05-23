import numpy as np
import robosuite as suite
from robosuite.controllers import load_controller_config
from robosuite.environments.manipulation.stack import Stack
from robosuite.models.objects import BoxObject
from robosuite.utils.placement_samplers import UniformRandomSampler
import gymnasium as gym
from gymnasium import spaces
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import time

class TowerBuildingEnv(gym.Env):
    def __init__(self, num_blocks=4, render=False):
        """
        Umgebung für einen Roboterarm, der lernt, einen stabilen Turm zu bauen
        
        Args:
            num_blocks: Anzahl der Blöcke für den Turm
            render: Ob die Umgebung visualisiert werden soll
        """
        super(TowerBuildingEnv, self).__init__()
        
        self.num_blocks = num_blocks
        self.render_mode = render
        
        # Controller konfigurieren
        self.controller_config = load_controller_config(default_controller="OSC_POSE")
        
        # Roboterumgebung initialisieren
        self.env = suite.make(
            env_name="Stack",
            robots="Panda",
            controller_configs=self.controller_config,
            has_renderer=render,
            has_offscreen_renderer=False,
            control_freq=20,
            horizon=500,
            initialization_noise={"magnitude": 0.1, "type": "uniform"},
            use_object_obs=True,
            use_camera_obs=False,
            reward_scale=1.0,
            placement_initializer=self._create_placement_initializer(),
            num_objects=num_blocks,
        )
        
        # Aktions- und Beobachtungsraum definieren
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(7,), dtype=np.float32
        )  # 6 DoF Roboterbewegung + Greifer
        
        # Beobachtungsraum: Roboterzustand + Objektpositionen + Orientierungen
        obs_dim = self.env.observation_spec()["robot0_proprio-state"].shape[0] + \
                  self.num_blocks * 7  # Position (3) + Quaternion (4) pro Block
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        
        self.episode_steps = 0
        self.max_episode_steps = 500
        self.previous_height = 0
        self.max_height_achieved = 0
        
    def _create_placement_initializer(self):
        """Erstellt einen Initializer, der Blöcke zufällig auf dem Tisch platziert"""
        pos_x_range = [-0.1, 0.1]
        pos_y_range = [-0.1, 0.1]
        
        # Rote Blöcke (alle die gleiche Größe)
        block_size = [0.04, 0.04, 0.04]
        block_density = 300
        
        # Liste mit BlockObject-Instanzen
        blocks = []
        for i in range(self.num_blocks):
            blocks.append(
                BoxObject(
                    name=f"block_{i}",
                    size=block_size,
                    rgba=[1, 0, 0, 1],  # Rote Blöcke
                    density=block_density,
                )
            )
        
        # Einen Sampler erstellen, der die Blöcke auf dem Tisch platziert
        placement_initializer = UniformRandomSampler(
            name="ObjectSampler",
            x_range=pos_x_range,
            y_range=pos_y_range,
            rotation=None,
            ensure_object_boundary_in_range=False,
            ensure_valid_placement=True,
            reference_pos=np.array((0, 0, 0.8)),
            z_offset=0.01,
        )
        
        # Blöcke zum Sampler hinzufügen
        for block in blocks:
            placement_initializer.add_objects(block)
        
        return placement_initializer
        
    def _get_obs(self):
        """Wandelt Robosuite-Beobachtungen in ein Format um, das für das RL-Training verwendet werden kann"""
        obs = self.env._get_observations()
        
        # Roboterzustand
        robot_state = obs["robot0_proprio-state"]
        
        # Objektinformationen (Position und Orientierung)
        obj_states = []
        for i in range(self.num_blocks):
            if f"block_{i}_pos" in obs:
                obj_pos = obs[f"block_{i}_pos"]
                obj_quat = obs[f"block_{i}_quat"]
                obj_states.extend(list(obj_pos) + list(obj_quat))
            else:
                # Fallback, wenn Objektinformationen fehlen
                obj_states.extend([0.0] * 7)
        
        # Kombinieren zu einem einzigen Vektor
        combined_obs = np.concatenate([robot_state, np.array(obj_states)])
        
        return combined_obs.astype(np.float32)
    
    def _calculate_tower_height(self):
        """Berechnet die Höhe des Turms basierend auf den Blockpositionen"""
        obs = self.env._get_observations()
        
        heights = []
        for i in range(self.num_blocks):
            if f"block_{i}_pos" in obs:
                # Z-Koordinate ist die Höhe
                heights.append(obs[f"block_{i}_pos"][2])
        
        # Die Turmhöhe ist der Abstand zwischen dem höchsten und dem niedrigsten Block
        if heights:
            tower_height = max(heights) - min(heights) + 0.04  # 0.04 ist die Blockgröße
        else:
            tower_height = 0.0
            
        return tower_height
    
    def _calculate_tower_stability(self):
        """Bewertet die Stabilität des Turms"""
        obs = self.env._get_observations()
        
        # Extrahiere die Positionen aller Blöcke
        positions = []
        for i in range(self.num_blocks):
            if f"block_{i}_pos" in obs:
                positions.append(obs[f"block_{i}_pos"])
        
        if len(positions) < 2:
            return 1.0  # Mit nur einem Block ist der Turm immer stabil
            
        # Sortiere Blöcke nach Höhe
        positions.sort(key=lambda pos: pos[2])
        
        # Berechne den Stabilitätsindex, basierend auf wie gut Blöcke übereinander gestapelt sind
        stability = 1.0
        for i in range(1, len(positions)):
            lower_block = positions[i-1]
            current_block = positions[i]
            
            # Horizontale Distanz zwischen Blöcken
            horizontal_dist = np.sqrt((current_block[0] - lower_block[0])**2 + 
                                     (current_block[1] - lower_block[1])**2)
            
            # Wenn die horizontale Distanz zu groß ist, wird der Turm instabil
            if horizontal_dist > 0.03:  # Schwellenwert für Instabilität
                stability *= 0.8  # Stabilitätsreduktion
                
        return stability
    
    def reset(self, seed=None):
        """Setzt die Umgebung zurück"""
        if seed is not None:
            np.random.seed(seed)
            
        self.env.reset()
        self.episode_steps = 0
        self.previous_height = 0
        self.max_height_achieved = 0
        
        return self._get_obs(), {}
    
    def step(self, action):
        """Führt eine Aktion in der Umgebung aus"""
        # Aktion auf den Bereich der Robosuite-Umgebung normalisieren
        action = np.clip(action, -1.0, 1.0)
        
        # Aktion in der Robosuite-Umgebung ausführen
        _, reward, done, _, info = self.env.step(action)
        
        # Zustand nach der Aktion abrufen
        obs = self._get_obs()
        
        # Eigene Belohnungsfunktion berechnen
        current_height = self._calculate_tower_height()
        stability = self._calculate_tower_stability()
        
        # Belohnung für Erhöhung des Turms
        height_reward = 0
        if current_height > self.max_height_achieved:
            height_reward = 2.0 * (current_height - self.max_height_achieved)
            self.max_height_achieved = current_height
        
        # Belohnung für Stabilität
        stability_reward = stability * 0.5
        
        # Kombinierte Belohnung
        custom_reward = height_reward + stability_reward
        
        # Bestrafung für zu lange Episoden
        self.episode_steps += 1
        if self.episode_steps >= self.max_episode_steps:
            done = True
        
        # Zusätzliche Informationen
        info = {
            "tower_height": current_height,
            "tower_stability": stability,
            "steps": self.episode_steps
        }
        
        # Konvertiere zu Gymnasium API (reward, terminated, truncated, info)
        terminated = done and self.episode_steps < self.max_episode_steps
        truncated = self.episode_steps >= self.max_episode_steps
        
        return obs, custom_reward, terminated, truncated, info
    
    def render(self):
        """Rendert die Umgebung, falls render_mode aktiviert ist"""
        if not self.render_mode:
            return
        
        self.env.render()
    
    def close(self):
        """Schließt die Umgebung"""
        self.env.close()


def train_tower_building_agent(total_timesteps=100000, render_training=False):
    """
    Trainiert einen Agenten zum Bauen von Türmen
    
    Args:
        total_timesteps: Gesamtanzahl der Trainingsschritte
        render_training: Ob der Trainingsprozess visualisiert werden soll
    
    Returns:
        Trainiertes PPO-Modell
    """
    # Umgebung erstellen
    env = TowerBuildingEnv(num_blocks=4, render=render_training)
    
    # Umgebung für Stable Baselines vektorisieren
    env = DummyVecEnv([lambda: env])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)
    
    # PPO-Modell erstellen
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=0.0003,
        n_steps=1024,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        tensorboard_log="./tower_building_tensorboard/"
    )
    
    # Training
    print("Starte Training...")
    model.learn(total_timesteps=total_timesteps)
    print("Training abgeschlossen!")
    
    # Modell speichern
    model.save("tower_building_ppo")
    
    return model


def evaluate_agent(model_path=None, num_episodes=5, render=True):
    """
    Evaluiert einen trainierten Agenten
    
    Args:
        model_path: Pfad zum gespeicherten Modell (oder None für neues Training)
        num_episodes: Anzahl der Testszenarien
        render: Ob die Ausführung visualisiert werden soll
    """
    # Umgebung erstellen
    env = TowerBuildingEnv(num_blocks=4, render=render)
    
    # Modell laden oder trainieren
    if model_path:
        model = PPO.load(model_path)
        print(f"Modell geladen von {model_path}")
    else:
        model = train_tower_building_agent(total_timesteps=100000, render_training=False)
    
    # Performance evaluieren
    avg_height = 0
    avg_stability = 0
    
    for i in range(num_episodes):
        print(f"Episode {i+1}/{num_episodes}")
        obs, _ = env.reset()
        done = False
        truncated = False
        episode_reward = 0
        
        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            
            if render:
                env.render()
                time.sleep(0.01)
        
        print(f"Episode {i+1} abgeschlossen:")
        print(f"- Turmhöhe: {info['tower_height']:.3f}")
        print(f"- Turmstabilität: {info['tower_stability']:.3f}")
        print(f"- Episoden-Belohnung: {episode_reward:.2f}")
        print("-" * 30)
        
        avg_height += info['tower_height']
        avg_stability += info['tower_stability']
    
    avg_height /= num_episodes
    avg_stability /= num_episodes
    
    print("\nErgebnisse der Evaluation:")
    print(f"Durchschnittliche Turmhöhe: {avg_height:.3f}")
    print(f"Durchschnittliche Turmstabilität: {avg_stability:.3f}")
    
    env.close()


if __name__ == "__main__":
    # Kurzes Training und Evaluation (für längeres Training total_timesteps erhöhen)
    model = train_tower_building_agent(total_timesteps=50000, render_training=False)
    
    # Trainierten Agenten evaluieren
    evaluate_agent(model_path=None, num_episodes=3, render=True)