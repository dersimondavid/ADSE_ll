import gym
import numpy as np
import pybullet as p
import pybullet_data
from gym import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
import matplotlib.pyplot as plt
import time
import os
from datetime import datetime
import zipfile

class RobotArmPickPlaceEnv(gym.Env):
    """
    Roboterarm Environment zum Greifen eines Würfels und Platzieren in einer Kiste
    """
    
    def __init__(self, gui=True):
        super(RobotArmPickPlaceEnv, self).__init__()
        
        self.gui = gui
        self.time_step = 1./240.
        
        # Aktionsraum: 7 DOF für Arm + 1 für Greifer
        self.action_space = spaces.Box(
            low=-1.0, 
            high=1.0, 
            shape=(8,), 
            dtype=np.float32
        )
        
        # Observationsraum: Arm-Positionen, Würfel-Position, Ziel-Position, Greifer-Zustand
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(20,), 
            dtype=np.float32
        )
        
        # Zielposition: 25cm über der Kiste
        self.target_position = np.array([0.5, 0.0, 0.5])  # Mittig über der Kiste
        
        # Physik initialisieren
        self._init_physics()
        
        # Episode Variablen
        self.episode_step = 0
        self.max_episode_steps = 6000
        self.cube_grasped = False
        self.success = False
        
    def _init_physics(self):
        """Initialisiert die PyBullet Physik-Simulation"""
        if self.gui:
            self.physics_client = p.connect(p.GUI)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
        else:
            self.physics_client = p.connect(p.DIRECT)
            
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(self.time_step)
        
        # Lade Objekte
        self._load_objects()
        
    def _load_objects(self):
        """Lädt alle Objekte in die Simulation"""
        # Boden
        self.plane_id = p.loadURDF("plane.urdf")
        
        # Roboterarm (Franka Panda)
        self.robot_id = p.loadURDF("franka_panda/panda.urdf", 
                                   basePosition=[0, 0, 0], 
                                   useFixedBase=True)
        
        # Würfel (0.04 Größe)
        cube_start_pos = [0.6, 0.2, 0.05]
        cube_start_orientation = p.getQuaternionFromEuler([0, 0, 0])
        self.cube_id = p.loadURDF("cube.urdf", 
                                  basePosition=cube_start_pos, 
                                  baseOrientation=cube_start_orientation,
                                  globalScaling=0.04)
        
        # Kiste (Tray)
        tray_pos = [0.5, 0.0, 0.02]
        self.tray_id = p.loadURDF("tray/traybox.urdf", 
                                  basePosition=tray_pos, 
                                  baseOrientation=p.getQuaternionFromEuler([0, 0, 0]))
          # Roboter-Joint-Informationen
        self.num_joints = p.getNumJoints(self.robot_id)
        self.joint_indices = list(range(7))  # Ersten 7 Joints für Arm
        self.gripper_indices = [9, 10]  # Greifer Joints
        
        # Setze initiale Arm-Position
        self._reset_robot_position()
        
    def _reset_robot_position(self):
        """Setzt den Roboter in eine neutrale Position"""
        initial_joint_positions = [0, -0.785, 0, -2.356, 0, 1.571, 0.785]
        
        for i, pos in enumerate(initial_joint_positions):
            p.resetJointState(self.robot_id, i, pos)
            
        # Greifer öffnen
        for joint_idx in self.gripper_indices:
            p.resetJointState(self.robot_id, joint_idx, 0.04)
            
    def reset(self):
        """Reset der Environment"""
        # Würfel hinter dem Roboterarm spawnen (negative X-Richtung)
        # Roboter ist bei [0,0,0], Kiste bei [0.5, 0.0], also Würfel bei negativen X-Werten
        cube_x = np.random.uniform(-0.4, -0.1)  # Hinter dem Roboter
        cube_y = np.random.uniform(-0.3, 0.3)   # Links/rechts variieren
        cube_pos = [cube_x, cube_y, 0.05]
        
        p.resetBasePositionAndOrientation(
            self.cube_id, 
            cube_pos, 
            p.getQuaternionFromEuler([0, 0, 0])
        )
        
        # Roboter zurücksetzen
        self._reset_robot_position()
        
        # Episode-Variablen zurücksetzen
        self.episode_step = 0
        self.cube_grasped = False
        self.success = False
        
        # Physik stabilisieren
        for _ in range(50):p
            p.stepSimulation()
            
        return self._get_observation()
    
    def _get_observation(self):
        """Erstellt die Observation für das RL-Modell"""
        # Arm Joint-Positionen
        joint_states = []
        for i in self.joint_indices:
            joint_state = p.getJointState(self.robot_id, i)[0]
            joint_states.append(joint_state)
            
        # Greifer-Position
        gripper_states = []
        for i in self.gripper_indices:
            gripper_state = p.getJointState(self.robot_id, i)[0]
            gripper_states.append(gripper_state)
            
        # End-Effector Position
        end_effector_pos = p.getLinkState(self.robot_id, 11)[0]
        
        # Würfel Position
        cube_pos, _ = p.getBasePositionAndOrientation(self.cube_id)
        
        # Ziel Position
        target_pos = self.target_position
        
        # Distanz zum Würfel und zum Ziel
        dist_to_cube = np.linalg.norm(np.array(end_effector_pos) - np.array(cube_pos))
        dist_to_target = np.linalg.norm(np.array(end_effector_pos) - np.array(target_pos))
        
        observation = np.concatenate([
            joint_states,           # 7 Werte
            gripper_states,         # 2 Werte
            end_effector_pos,       # 3 Werte
            cube_pos,              # 3 Werte
            target_pos,            # 3 Werte
            [dist_to_cube],        # 1 Wert
            [dist_to_target]       # 1 Wert
        ]).astype(np.float32)
        
        return observation
    
    def step(self, action):
        """Führt einen Schritt in der Environment aus"""
        self.episode_step += 1
        
        # Aktionen normalisieren und anwenden
        arm_actions = action[:7] * 0.05  # Langsame Bewegung
        gripper_action = action[7]
        
        # Arm bewegen
        current_joint_positions = []
        for i in self.joint_indices:
            current_pos = p.getJointState(self.robot_id, i)[0]
            new_pos = current_pos + arm_actions[i]
            current_joint_positions.append(new_pos)
            
        # Joint-Limits einhalten
        for i, pos in enumerate(current_joint_positions):
            p.setJointMotorControl2(
                self.robot_id, 
                i, 
                p.POSITION_CONTROL, 
                targetPosition=pos,
                maxVelocity=0.5  # Langsame Bewegung
            )
        
        # Greifer steuern
        if gripper_action > 0:
            gripper_pos = 0.04  # Geöffnet
        else:
            gripper_pos = 0.0   # Geschlossen
            
        for joint_idx in self.gripper_indices:
            p.setJointMotorControl2(
                self.robot_id, 
                joint_idx, 
                p.POSITION_CONTROL, 
                targetPosition=gripper_pos
            )
        
        # Physik-Schritt
        p.stepSimulation()
        if self.gui:
            time.sleep(self.time_step)
            
        # Neue Observation
        obs = self._get_observation()
        
        # Belohnung berechnen
        reward = self._calculate_reward()
        
        # Episode beendet?
        done = self._check_done()
        
        info = {
            'cube_grasped': self.cube_grasped,
            'success': self.success,
            'episode_step': self.episode_step
        }
        
        return obs, reward, done, info
    
    def _calculate_reward(self):
        """Berechnet die Belohnung"""
        reward = 0.0
        
        # Aktuelle Positionen
        end_effector_pos = np.array(p.getLinkState(self.robot_id, 11)[0])
        cube_pos, _ = p.getBasePositionAndOrientation(self.cube_id)
        cube_pos = np.array(cube_pos)
        
        # Prüfe ob Würfel gegriffen wurde
        contacts = p.getContactPoints(self.robot_id, self.cube_id)
        cube_grasped = len(contacts) > 0 and cube_pos[2] > 0.1
        
        if cube_grasped and not self.cube_grasped:
            reward += 50.0  # Belohnung für erstmaliges Greifen
            self.cube_grasped = True
            
        if self.cube_grasped:
            # Belohnung für Bewegung zum Ziel
            dist_to_target = np.linalg.norm(end_effector_pos - self.target_position)
            reward += -dist_to_target * 10.0
            
            # Prüfe ob an Zielposition
            if dist_to_target < 0.05:
                reward += 100.0  # Große Belohnung für Erreichen des Ziels
                
                # Automatisches Öffnen des Greifers am Ziel
                for joint_idx in self.gripper_indices:
                    p.setJointMotorControl2(
                        self.robot_id, 
                        joint_idx, 
                        p.POSITION_CONTROL, 
                        targetPosition=0.04
                    )
                
                # Prüfe ob Würfel in Kiste gefallen ist
                if cube_pos[2] < 0.15 and np.linalg.norm(cube_pos[:2] - [0.5, 0.0]) < 0.1:
                    reward += 200.0  # Erfolg!
                    self.success = True
        else:
            # Belohnung für Annäherung an Würfel
            dist_to_cube = np.linalg.norm(end_effector_pos - cube_pos)
            reward += -dist_to_cube * 5.0
            
        # Kleine Strafe für Zeit
        reward -= 0.05
        
        return reward
    
    def _check_done(self):
        """Prüft ob Episode beendet ist"""
        return (self.episode_step >= self.max_episode_steps or 
                self.success)
    
    def close(self):
        """Schließt die Environment"""
        p.disconnect(self.physics_client)


class TrainingCallback(BaseCallback):
    """Callback für das Training mit Plotting und Model-Speicherung"""
    
    def __init__(self, save_freq=20000, save_path='./models/', verbose=1):
        super(TrainingCallback, self).__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_episode_reward = 0
        self.current_episode_length = 0
        
        # Erstelle Ordner
        os.makedirs(save_path, exist_ok=True)
        
    def _on_step(self) -> bool:
        self.current_episode_reward += self.locals['rewards'][0]
        self.current_episode_length += 1
        
        # Episode beendet?
        if self.locals['dones'][0]:
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_lengths.append(self.current_episode_length)
            self.current_episode_reward = 0
            self.current_episode_length = 0
            
        # Model speichern
        if self.n_calls % self.save_freq == 0:
            model_path = os.path.join(
                self.save_path, 
                f'model_step_{self.n_calls}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.zip'
            )
            self.model.save(model_path)
            print(f"Model gespeichert: {model_path}")
            
            # Plot erstellen
            self._create_training_plot()
            
        return True
    
    def _create_training_plot(self):
        """Erstellt Training-Plots"""
        if len(self.episode_rewards) < 10:
            return
            
        plt.figure(figsize=(12, 4))
        
        # Belohnungs-Plot
        plt.subplot(1, 2, 1)
        plt.plot(self.episode_rewards[-1000:])  # Letzte 1000 Episoden
        plt.title('Episode Rewards (Last 1000)')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.grid(True)
        
        # Episoden-Längen Plot
        plt.subplot(1, 2, 2)
        plt.plot(self.episode_lengths[-1000:])  # Letzte 1000 Episoden
        plt.title('Episode Lengths (Last 1000)')
        plt.xlabel('Episode')
        plt.ylabel('Steps')
        plt.grid(True)
        
        plt.tight_layout()
        
        # Plot speichern
        plot_path = os.path.join(
            self.save_path, 
            f'training_progress_{self.n_calls}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        )
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Training-Plot gespeichert: {plot_path}")


def train_model(gui=True, total_timesteps=500000):
    """Trainiert das RL-Modell"""
    print(f"Starte Training {'mit' if gui else 'ohne'} GUI...")
    
    # Environment erstellen
    env = RobotArmPickPlaceEnv(gui=gui)
    
    # Model erstellen
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        tensorboard_log="./ppo_pickplace_tensorboard/"
    )
    
    # Callback für Speicherung und Plotting
    callback = TrainingCallback(save_freq=20000, save_path='./training_logs/')
    
    # Training starten
    model.learn(
        total_timesteps=total_timesteps,
        callback=callback,
        tb_log_name=f"PPO_PickPlace_{'GUI' if gui else 'NoGUI'}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    
    # Finales Model speichern
    final_model_path = f"final_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
    model.save(final_model_path)
    print(f"Training beendet. Finales Model gespeichert: {final_model_path}")
    
    env.close()
    return model, final_model_path


def test_model(model_path, gui=True, episodes=10):
    """Testet ein trainiertes Modell"""
    print(f"Teste Model: {model_path}")
    
    # Environment erstellen
    env = RobotArmPickPlaceEnv(gui=gui)
    
    # Model laden
    model = PPO.load(model_path)
    
    success_count = 0
    
    for episode in range(episodes):
        obs = env.reset()
        episode_reward = 0
        done = False
        
        print(f"Episode {episode + 1}/{episodes}")
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            episode_reward += reward
            
            if gui:
                time.sleep(0.01)  # Etwas verlangsamen für bessere Sichtbarkeit
                
        if info['success']:
            success_count += 1
            
        print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}, Success = {info['success']}")
    
    success_rate = success_count / episodes * 100
    print(f"\nErgebnis: {success_count}/{episodes} erfolgreiche Episoden ({success_rate:.1f}%)")
    
    env.close()


def select_model_file():
    """Zeigt alle verfügbaren .zip Modell-Dateien an und lässt den Benutzer eine auswählen"""
    # Sammle alle .zip Dateien im aktuellen Verzeichnis und Unterordnern
    zip_files = []
    
    # Aktuelles Verzeichnis
    for file in os.listdir('.'):
        if file.endswith('.zip'):
            zip_files.append(file)
    
    # training_logs Ordner
    if os.path.exists('./training_logs/'):
        for file in os.listdir('./training_logs/'):
            if file.endswith('.zip'):
                zip_files.append(f'./training_logs/{file}')
    
    # Wenn keine .zip Dateien gefunden wurden
    if not zip_files:
        print("❌ Keine .zip Modell-Dateien gefunden!")
        print("Tipp: Trainieren Sie zuerst ein Modell oder geben Sie den vollständigen Pfad ein.")
        model_path = input("Vollständigen Pfad zur .zip Datei eingeben: ").strip()
        return model_path if model_path else None
    
    # Sortiere Dateien nach Änderungsdatum (neueste zuerst)
    zip_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    
    # Zeige verfügbare Modelle an
    print("\n📁 Verfügbare Modell-Dateien:")
    print("=" * 60)
    for i, file in enumerate(zip_files, 1):
        # Dateigröße und Änderungsdatum anzeigen
        file_size = os.path.getsize(file) / (1024 * 1024)  # MB
        mod_time = datetime.fromtimestamp(os.path.getmtime(file)).strftime('%d.%m.%Y %H:%M')
        print(f"{i:2d}. {os.path.basename(file)}")
        print(f"    📍 {file}")
        print(f"    📊 {file_size:.1f} MB | 📅 {mod_time}")
        print()
    
    print(f"{len(zip_files) + 1:2d}. ✏️  Eigenen Pfad eingeben")
    print("=" * 60)
    
    while True:
        try:
            choice = input(f"\nWählen Sie ein Modell (1-{len(zip_files) + 1}): ").strip()
            choice_num = int(choice)
            
            if 1 <= choice_num <= len(zip_files):
                selected_file = zip_files[choice_num - 1]
                print(f"✅ Ausgewählt: {os.path.basename(selected_file)}")
                return selected_file
            elif choice_num == len(zip_files) + 1:
                model_path = input("Vollständigen Pfad zur .zip Datei eingeben: ").strip()
                return model_path if model_path else None
            else:
                print(f"❌ Ungültige Eingabe! Bitte wählen Sie 1-{len(zip_files) + 1}")
        except ValueError:
            print("❌ Bitte geben Sie eine gültige Nummer ein!")


def main():
    """Hauptfunktion mit Menü"""
    print("=== Roboterarm Pick-and-Place RL Training ===")
    print("1. Training mit GUI")
    print("2. Training ohne GUI")
    print("3. Modell testen")
    print("4. Beenden")
    
    while True:
        choice = input("\nWählen Sie eine Option (1-4): ").strip()
        
        if choice == '1':
            timesteps = input("Anzahl Trainingsschritte (Standard: 500000): ").strip()
            timesteps = int(timesteps) if timesteps else 500000
            train_model(gui=True, total_timesteps=timesteps)
        elif choice == '2':
            timesteps = input("Anzahl Trainingsschritte (Standard: 500000): ").strip()
            timesteps = int(timesteps) if timesteps else 500000
            train_model(gui=False, total_timesteps=timesteps)
            
        elif choice == '3':
            model_path = select_model_file()
            if not model_path or not os.path.exists(model_path):
                print("❌ Modell-Datei nicht gefunden oder Auswahl abgebrochen!")
                continue
                
            gui_choice = input("Mit GUI testen? (j/n): ").strip().lower()
            gui = gui_choice == 'j'
            
            episodes = input("Anzahl Test-Episoden (Standard: 10): ").strip()
            episodes = int(episodes) if episodes else 10
            
            test_model(model_path, gui=gui, episodes=episodes)
            
        elif choice == '4':
            print("Programm beendet.")
            break
            
        else:
            print("Ungültige Eingabe. Bitte wählen Sie 1-4.")


if __name__ == "__main__":
    main()