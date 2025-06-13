# SAC + HER Pick-and-Place Environment
# Neue Kombination aus SAC (kontinuierliche Kontrolle) + HER (Goal-Relabeling)

import os
import sys
import math
import time
import numpy as np
import matplotlib.pyplot as plt
import torch

# PyBullet fr Physik-Simulation
import pybullet as p
import pybullet_data

# Gymnasium fr Environment-Interface
import gymnasium as gym
from gymnasium import spaces

# Stable Baselines3 fr SAC und HER
from stable_baselines3 import SAC
from stable_baselines3.her import HerReplayBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from datetime import datetime

class RobotArmPickPlaceEnvHER(gym.Env):
    def __init__(self, gui=True, debug_mode=False):
        super(RobotArmPickPlaceEnvHER, self).__init__()
        
        self.gui = gui
        self.debug_mode = debug_mode
        self.time_step = 1./240.
        
        # Aktionsraum: 4D kontinuierliche Kontrolle
        self.action_space = spaces.Box(
            low=-1.0, 
            high=1.0, 
            shape=(4,),  # XYZ-Movement + Gripper
            dtype=np.float32
        )        # GOAL-BASIERTE Observation (fr HER erforderlich) - ERWEITERT
        obs_dim = 24  # Vollstndige Observation: 3+3+4+3+3+3+3+2 = 24 (inkl. Kontakt-Info)
        goal_dim = 5   # Erweiterte Goals: [x, y, z, is_grasped, gripper_distance]
        
        self.observation_space = spaces.Dict({
            'observation': spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32),
            'achieved_goal': spaces.Box(low=-np.inf, high=np.inf, shape=(goal_dim,), dtype=np.float32),
            'desired_goal': spaces.Box(low=-np.inf, high=np.inf, shape=(goal_dim,), dtype=np.float32),
        })
        
        # Festes Ziel - die Kiste
        self.target_position = np.array([0.5, 0.0, 0.08])
        
        # HER-Parameter
        self.distance_threshold = 0.1  # Erfolgsschwelle: 10cm Nhe zum Ziel
        
        # Physik initialisieren
        self._init_physics()
          # Episode Variablen
        self.episode_step = 0
        self.max_episode_steps = 1000  # REDUZIERT fr schnelleres HER-Lernen
        self.cube_grasped = False
        self.success = False
        self.grasp_constraint = None        #  ADAPTIVE EXPLORATION & ANTI-LOKALE-OPTIMA SYSTEM
        self.exploration_boost = 0.0  # Zustzliche Exploration bei Stagnation
        self.stagnation_detection = {
            'no_improvement_steps': 0,
            'best_distance_to_cube': float('inf'),
            'best_cube_to_target': float('inf'),
            'steps_without_grasp_attempt': 0,
            'exploration_boost_active': False,
            'forced_action_counter': 0,            'hovering_steps': 0,  # Zhlt Steps ber dem Wrfel ohne Absenkung
            'hovering_threshold': 5,  # REDUZIERT: Frhere Erkennung fr strkeren Lerndruck - reines RL-Training
            'descent_motivation_active': False  # Zeigt an ob Absenkungsmotivation aktiv ist (nur fr Monitoring)
        }
        
        #  HAUPTZIEL-ORIENTIERTES LERNEN
        self.main_goal_weight = 1.0  # Gewichtung des Hauptziels
        self.sub_goal_decay = 0.95   # Abbau der Unterziel-Belohnungen ber Zeit
        self.episodes_since_success = 0
        self.urgency_factor = 1.0    # Steigt bei wiederholten Fehlschlgen
        
        # Performance Tracking fr Debugging
        self.last_distance = float('inf')
        
        # PHASEN-BASIERTE BELOHNUNGSSTRUKTUR
        self.last_gripper_to_cube_dist = float('inf')
        self.last_cube_to_target_dist = float('inf')
        self.phase_bonuses = {
            'exploration': 0.0,
            'approach': 0.0,
            'grasp': 0.0,
            'transport': 0.0,
            'release': 0.0
        }
        self.consecutive_improvements = 0
        self.stagnation_counter = 0
        
        # CURRICULUM LEARNING SYSTEM fr progressive Schwierigkeit
        self.training_phase = 0  # 0=Easy, 1=Medium, 2=Hard, 3=Expert
        self.success_streak = 0
        self.phase_success_threshold = [0.3, 0.5, 0.7, 0.9]  # Erfolgsraten fr Phasen-Upgrade
        self.recent_success_rate = 0.0
        
    def _init_physics(self):
        # Initialisiere PyBullet Physik-Engine
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
        # Boden
        self.plane_id = p.loadURDF("plane.urdf")
        
        # Roboterarm (Franka Panda)
        self.robot_id = p.loadURDF("franka_panda/panda.urdf", 
                                   basePosition=[0, 0, 0], 
                                   useFixedBase=True,
                                   flags=p.URDF_USE_SELF_COLLISION)
        
        # Wrfel (0.04 Gre)
        self.cube_id = p.loadURDF("cube.urdf", 
                                  basePosition=[0, 0, 0], 
                                  baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
                                  globalScaling=0.04)
        
        # Kiste (Tray) - FESTE Position
        tray_pos = [0.5, 0.0, 0.02]
        self.tray_id = p.loadURDF("tray/traybox.urdf", 
                                  basePosition=tray_pos, 
                                  baseOrientation=p.getQuaternionFromEuler([0, 0, 0]))
        
        # Roboter-Joint-Informationen
        self.num_joints = p.getNumJoints(self.robot_id)
        self.arm_joint_indices = list(range(7))  # Joints 0-6 fr den Arm
        self.gripper_joint_indices = [9, 10]    # Joints 9, 10 fr Greifer-Finger
        self.end_effector_link_index = 11        # Link 11 ist der End-Effector
        
        # Setze initiale Arm-Position
        self._reset_robot_position()
        
        # Markiere Zielposition visuell (nur bei GUI)
        if self.gui:
            p.addUserDebugLine([0.5, 0.0, 0.0], [0.5, 0.0, 0.5], [1, 0, 0], lineWidth=3)
            p.addUserDebugText("TARGET", [0.5, 0.0, 0.4], [1, 0, 0], textSize=2)
        
    def _reset_robot_position(self):
        # Neutral-Position
        neutral_angles = [0.0, -0.3135, 0.0, -2.515, 0.0, 2.226, 0.87]
        
        # ALLE Joints zurcksetzen
        for i in range(self.num_joints):
            if i < 7:  # Arm-Joints (0-6)
                p.resetJointState(self.robot_id, i, neutral_angles[i])
            elif i in [9, 10]:  # Greifer-Finger (9, 10)
                p.resetJointState(self.robot_id, i, 0.04)  # Voll geffnet
            else:
                p.resetJointState(self.robot_id, i, 0.0)
        
        # Motor-Control fr Arm
        for i in range(7):
            p.setJointMotorControl2(
                self.robot_id, i, p.POSITION_CONTROL,
                targetPosition=neutral_angles[i], 
                force=200,
                maxVelocity=1.0
            )
        
        # Greifer-Motoren
        for joint_idx in [9, 10]:
            p.setJointMotorControl2(
                self.robot_id, joint_idx, p.POSITION_CONTROL,                targetPosition=0.04,  # Voll geffnet
                force=50,
                maxVelocity=0.5
            )
    
    def reset(self, seed=None, options=None):
        # Seed fr Reproduzierbarkeit
        if seed is not None:
            np.random.seed(seed)

        # MEMORY LEAK PREVENTION: Alte Constraints entfernen
        try:
            if hasattr(self, 'grasp_constraint') and self.grasp_constraint is not None:
                p.removeConstraint(self.grasp_constraint)
        except Exception:
            pass  # Constraint existiert mglicherweise nicht mehr
        finally:
            self.grasp_constraint = None
              #  ADAPTIVE CURRICULUM: Wrfel-Position basierend auf Lernerfolg
        cube_x, cube_y = self._get_adaptive_cube_position()
        cube_z = 0.05
        
        if self.debug_mode:
            print(f" EPISODE RESET - Phase {self.training_phase} - Wrfel bei [{cube_x:.3f}, {cube_y:.3f}]")
        
        # Wrfel positionieren
        p.resetBasePositionAndOrientation(
            self.cube_id, 
            [cube_x, cube_y, cube_z], 
            p.getQuaternionFromEuler([0, 0, np.random.uniform(0, 2*np.pi)])
        )
        
        # Roboter zurcksetzen
        self._reset_robot_position()
        
        # Physik stabilisieren
        for _ in range(100):
            p.stepSimulation()
            if self.gui:
                time.sleep(1./240.)
          # Episode-Variablen zurcksetzen
        self.episode_step = 0
        self.cube_grasped = False
        self.success = False
        if self.grasp_constraint is not None:
            p.removeConstraint(self.grasp_constraint)
        self.grasp_constraint = None
          # Performance Tracking zurcksetzen (ERWEITERT fr Anti-Lokale-Optima)
        self.last_distance = float
        self.last_gripper_distance = float('inf')  # Fr Verbesserungs-Tracking
        self.last_transport_distance = float('inf')  # Fr Transport-Verbesserung
        self.steps_near_cube = 0  # Anti-Camping Counter
          # PHASEN-TRACKING ZURCKSETZEN
        self.last_gripper_to_cube_dist = float('inf')
        self.last_cube_to_target_dist = float('inf')
        self.phase_bonuses = {
            'exploration': 0.0,
            'approach': 0.0,
            'grasp': 0.0,
            'transport': 0.0,
            'release': 0.0
        }
        self.consecutive_improvements = 0
        self.stagnation_counter = 0
          # CURRICULUM LEARNING INITIALISIERUNG (nur beim ersten Reset)
        if not hasattr(self, 'training_phase'):
            self.training_phase = 0
            self.success_streak = 0
            self.phase_success_threshold = [0.3, 0.5, 0.7, 0.9]
            self.recent_success_rate = 0.0
            self.recent_successes = []
        
        obs = self._get_obs()
        info = {'is_success': False}  # Zu Beginn noch kein Erfolg
        
        # NEU: Event-basierte Belohnungs-Tracker zurücksetzen
        self.was_in_auto_grasp_zone = False
        self.contact_achieved = False
        self.last_xy_distance = float('inf')
        self.last_z_pos = None
        self.last_close_distance = float('inf')
        
        return obs, info
    def _get_obs(self):
        try:
            # 1. End-Effector Position (3 Werte)
            end_effector_state = p.getLinkState(self.robot_id, self.end_effector_link_index)
            end_effector_pos = np.array(end_effector_state[0])
            
            # NaN-Check fr End-Effector
            if np.any(np.isnan(end_effector_pos)) or np.any(np.isinf(end_effector_pos)):
                print(f" NaN/Inf in end_effector_pos: {end_effector_pos}")
                end_effector_pos = np.array([0.3, 0.0, 0.4])  # Safe fallback
            
            # 2. Greifer-Zustand (2 Werte)
            try:
                gripper_pos_left = p.getJointState(self.robot_id, 9)[0]
                gripper_pos_right = p.getJointState(self.robot_id, 10)[0]
                
                # NaN-Check fr Greifer
                if np.isnan(gripper_pos_left) or np.isnan(gripper_pos_right):
                    print(f" NaN in gripper positions: left={gripper_pos_left}, right={gripper_pos_right}")
                    gripper_pos_left = gripper_pos_right = 0.03  # Safe fallback
                    
                avg_gripper_pos = (gripper_pos_left + gripper_pos_right) / 2.0
                
            except Exception as e:
                print(f" Greifer-Zustand Fehler: {e}")
                gripper_pos_left = gripper_pos_right = avg_gripper_pos = 0.03
            
            # 3. Wrfel Position und Orientierung (4 Werte)
            try:
                cube_pos, cube_orn = p.getBasePositionAndOrientation(self.cube_id)
                cube_pos = np.array(cube_pos)
                
                # NaN-Check fr Wrfel
                if np.any(np.isnan(cube_pos)) or np.any(np.isinf(cube_pos)):
                    print(f" NaN/Inf in cube_pos: {cube_pos}")
                    cube_pos = np.array([0.0, 0.0, 0.05])  # Safe fallback
                
                # Euler-Winkel fr kompakte Orientierung (nur Z-Rotation ist relevant)
                cube_euler = p.getEulerFromQuaternion(cube_orn)
                cube_z_rotation = cube_euler[2]  # Rotation um Z-Achse
                
                if np.isnan(cube_z_rotation) or np.isinf(cube_z_rotation):
                    cube_z_rotation = 0.0
                    
            except Exception as e:
                print(f" Wrfel-Position Fehler: {e}")
                cube_pos = np.array([0.0, 0.0, 0.05])
                cube_z_rotation = 0.0
            
            # 4. Relative Vektoren (6 Werte) - mit NaN-Schutz
            end_effector_to_cube = cube_pos - end_effector_pos
            end_effector_to_target = self.target_position - end_effector_pos
            cube_to_target = self.target_position - cube_pos
            
            # NaN-Check fr relative Vektoren
            for vec_name, vec in [("end_effector_to_cube", end_effector_to_cube), 
                                 ("end_effector_to_target", end_effector_to_target),
                                 ("cube_to_target", cube_to_target)]:
                if np.any(np.isnan(vec)) or np.any(np.isinf(vec)):
                    print(f" NaN/Inf in {vec_name}: {vec}")
                    # Recalculate with safe values
                    if vec_name == "end_effector_to_cube":
                        end_effector_to_cube = np.array([0.0, 0.0, -0.3])
                    elif vec_name == "end_effector_to_target":
                        end_effector_to_target = np.array([0.2, 0.0, -0.3])
                    else:  # cube_to_target
                        cube_to_target = np.array([0.5, 0.0, 0.03])
              # 5. KONTAKT-INFORMATIONEN (3 Werte) - WICHTIG FR GREIFEN!
            try:
                contacts_left = p.getContactPoints(bodyA=self.robot_id, bodyB=self.cube_id, linkIndexA=9)
                contacts_right = p.getContactPoints(bodyA=self.robot_id, bodyB=self.cube_id, linkIndexA=10)
                
                contact_left = 1.0 if len(contacts_left) > 0 else 0.0
                contact_right = 1.0 if len(contacts_right) > 0 else 0.0
                bilateral_contact = 1.0 if (contact_left > 0.5 and contact_right > 0.5) else 0.0
                
            except Exception:
                contact_left = 0.0
                contact_right = 0.0
                bilateral_contact = 0.0
            
            # 6. Task-Status (2 Werte)
            task_progress = float(self.episode_step) / self.max_episode_steps
            gripper_to_cube_dist = np.linalg.norm(end_effector_to_cube)
            
            # Final NaN-Check fr Distanz
            if np.isnan(gripper_to_cube_dist) or np.isinf(gripper_to_cube_dist):
                gripper_to_cube_dist = 1.0  # Safe fallback
            
            # Observation zusammensetzen (3+3+4+3+3+3+3+2 = 24 Werte)
            observation = np.concatenate([
                end_effector_pos,           # 3 - End-Effector Position 
                [gripper_pos_left, gripper_pos_right, avg_gripper_pos],  # 3 - Detaillierte Greifer-Info
                cube_pos, [cube_z_rotation], # 4 - Wrfel Position + Orientierung
                end_effector_to_cube,       # 3 - Vektor: Greifer -> Wrfel
                end_effector_to_target,     # 3 - Vektor: Greifer -> Ziel (WICHTIG!)
                cube_to_target,             # 3 - Vektor: Wrfel -> Ziel
                [contact_left, contact_right, bilateral_contact], # 3 - KONTAKT-INFO!
                [task_progress, gripper_to_cube_dist]  # 2 - Task Status + Distanz
            ]).astype(np.float32)
            
            # KRITISCHER NaN-Check fr finale Observation
            if np.any(np.isnan(observation)) or np.any(np.isinf(observation)):
                print(f" KRITISCH: NaN/Inf in finaler Observation!")
                print(f"Observation: {observation}")
                # Emergency fallback - komplett neue, sichere observation
                observation = np.zeros(21, dtype=np.float32)
                observation[:3] = [0.3, 0.0, 0.4]  # Safe end effector
                observation[3:6] = [0.03, 0.03, 0.03]  # Safe gripper
                observation[6:9] = [0.0, 0.0, 0.05]  # Safe cube pos
                observation[9] = 0.0  # Safe rotation
                observation[10:13] = [-0.3, 0.0, -0.35]  # Safe vectors
                observation[13:16] = [0.2, 0.0, -0.35]
                observation[16:19] = [0.5, 0.0, 0.03]
                observation[19] = task_progress if not np.isnan(task_progress) else 0.0
                observation[20] = 1.0  # Safe distance
            
            # HER-Format: 
            # achieved_goal = [cube_x, cube_y, cube_z, is_grasped_binary, gripper_to_cube_distance]
            is_grasped_binary = 1.0 if self.cube_grasped else 0.0
            achieved_goal = np.array([
                cube_pos[0], cube_pos[1], cube_pos[2],  # Wrfel Position
                is_grasped_binary,                       # Greif-Status (0/1)
                gripper_to_cube_dist                     # Distanz Greifer->Wrfel
            ]).astype(np.float32)
            
            # NaN-Check fr achieved_goal
            if np.any(np.isnan(achieved_goal)) or np.any(np.isinf(achieved_goal)):
                print(f" NaN/Inf in achieved_goal: {achieved_goal}")
                achieved_goal = np.array([0.0, 0.0, 0.05, 0.0, 1.0], dtype=np.float32)
            
            # desired_goal = [target_x, target_y, target_z, should_be_grasped, optimal_gripper_dist]
            desired_goal = np.array([
                self.target_position[0], self.target_position[1], self.target_position[2],  # Ziel-Position
                1.0,  # Sollte gegriffen werden
                0.0   # Optimale Greifer-Distanz (0 = perfekt gegriffen)
            ]).astype(np.float32)
            
            # NaN-Check fr desired_goal
            if np.any(np.isnan(desired_goal)) or np.any(np.isinf(desired_goal)):
                print(f" NaN/Inf in desired_goal: {desired_goal}")
                desired_goal = np.array([0.5, 0.0, 0.08, 1.0, 0.0], dtype=np.float32)
            return {
                'observation': observation,
                'achieved_goal': achieved_goal,
                'desired_goal': desired_goal,
            }
            
        except Exception as e:
            print(f" KRITISCHER FEHLER in _get_obs: {e}")
            # Emergency fallback observation
            return {
                'observation': np.zeros(24, dtype=np.float32),  # Angepasst fr neue Dimension
                'achieved_goal': np.array([0.0, 0.0, 0.05, 0.0, 1.0], dtype=np.float32),
                'desired_goal': np.array([0.5, 0.0, 0.08, 1.0, 0.0], dtype=np.float32),
            }

    def _update_grasp_state_v2(self):
        if not hasattr(self, 'robot_id') or not hasattr(self, 'cube_id'):
            return
        
        try:
            # Position des Endeffektors
            link_state = p.getLinkState(self.robot_id, self.end_effector_link_index)
            end_effector_pos = np.array(link_state[0])
            
            # Position des Wrfels
            cube_pos, _ = p.getBasePositionAndOrientation(self.cube_id)
            cube_pos = np.array(cube_pos)
            
            # Distanz berechnen
            distance_to_cube = np.linalg.norm(end_effector_pos - cube_pos)
            
            # Greifer-Positionen abrufen
            gripper_state_left = p.getJointState(self.robot_id, 9)
            gripper_state_right = p.getJointState(self.robot_id, 10)
            gripper_pos_left = gripper_state_left[0]
            gripper_pos_right = gripper_state_right[0]
            avg_gripper_pos = (gripper_pos_left + gripper_pos_right) / 2.0
            
        except Exception as e:
            if self.debug_mode:
                print(f" Fehler in _update_grasp_state_v2: {e}")
            return
        
        # GREIFEN: Sehr nah am Wrfel UND Greifer deutlich geschlossen UND Kontakt
        if distance_to_cube < 0.06 and avg_gripper_pos < 0.015:
            if self.grasp_constraint is None:
                # Prfe strkeren Kontakt zwischen beiden Greifer-Fingern und Wrfel
                contacts_left = p.getContactPoints(bodyA=self.robot_id, bodyB=self.cube_id, linkIndexA=9)
                contacts_right = p.getContactPoints(bodyA=self.robot_id, bodyB=self.cube_id, linkIndexA=10)
                if len(contacts_left) > 0 or len(contacts_right) > 0:
                    # ROBUSTERE CONSTRAINT ERSTELLUNG
                    self.grasp_constraint = p.createConstraint(
                        parentBodyUniqueId=self.robot_id,
                        parentLinkIndex=self.end_effector_link_index,
                        childBodyUniqueId=self.cube_id,
                        childLinkIndex=-1,
                        jointType=p.JOINT_FIXED,
                        jointAxis=[0, 0, 0],
                        parentFramePosition=[0, 0, 0],
                        childFramePosition=[0, 0, 0]
                    )
                    self.cube_grasped = True
                    if self.debug_mode:
                        print(f" ERFOLGREICH GEGRIFFEN! Distanz: {distance_to_cube:.4f}")

        # LOSLASSEN: Greifer weit geffnet ODER strategisches Loslassen in der Kiste
        elif avg_gripper_pos > 0.025 and self.grasp_constraint is not None:
            # Prfe ob wir ber der Kiste sind fr kontrolliertes Loslassen
            cube_to_target_dist = np.linalg.norm(cube_pos - self.target_position)
            
            if cube_to_target_dist < 0.12:  # Nah genug an der Kiste
                p.removeConstraint(self.grasp_constraint)
                self.grasp_constraint = None
                self.cube_grasped = False
                if self.debug_mode:
                    print(" STRATEGISCHES LOSLASSEN in der Kiste!")
            else:
                # Notfall-Loslassen wenn Greifer zu weit geffnet
                p.removeConstraint(self.grasp_constraint)
                self.grasp_constraint = None
                self.cube_grasped = False
                if self.debug_mode:
                    print(" NOTFALL-LOSLASSEN: Greifer zu weit geffnet")
    
    # GoalEnv Interface fr HER-Kompatibilitt
    def goal_env_compute_reward(self, achieved_goal, desired_goal, info):
        return self.compute_reward(achieved_goal, desired_goal, info)
    
    def goal_env_is_success(self, achieved_goal, desired_goal):
        return self._is_success(achieved_goal, desired_goal)
    
    def close(self):
        try:
            # Alle Constraints entfernen
            if hasattr(self, 'grasp_constraint') and self.grasp_constraint is not None:
                p.removeConstraint(self.grasp_constraint)
                self.grasp_constraint = None
            
            # PyBullet-Verbindung sicher trennen
            if hasattr(self, 'physics_client'):
                p.disconnect(self.physics_client)
        except Exception as e:
            if self.debug_mode:
                print(f"Warning during environment close: {e}")
        finally:
            # Sicherstellen, dass Referenzen gelscht werden
            self.physics_client = None
    
    # GoalEnv Interface fr HER-Kompatibilitt
    def goal_env_compute_reward(self, achieved_goal, desired_goal, info):
        return self.compute_reward(achieved_goal, desired_goal, info)
    
    def goal_env_is_success(self, achieved_goal, desired_goal):
        return self._is_success(achieved_goal, desired_goal)
    
    def close(self):
        try:
            # Alle Constraints entfernen
            if hasattr(self, 'grasp_constraint') and self.grasp_constraint is not None:
                p.removeConstraint(self.grasp_constraint)
                self.grasp_constraint = None
            
            # PyBullet-Verbindung sicher trennen
            if hasattr(self, 'physics_client'):
                p.disconnect(self.physics_client)
        except Exception as e:
            if self.debug_mode:
                print(f"Warning during environment close: {e}")
        finally:
            # Sicherstellen, dass Referenzen gelscht werden
            self.physics_client = None
    
    def step(self, action):
        self.episode_step += 1
        
        # KRITISCHER NaN-Check fr Input-Action
        if not isinstance(action, np.ndarray):
            action = np.array(action, dtype=np.float32)
        
        if np.any(np.isnan(action)) or np.any(np.isinf(action)):
            print(f" KRITISCH: NaN/Inf in Action: {action}")
            action = np.zeros(4, dtype=np.float32)  # Safe fallback
        
        #  ADAPTIVE EXPLORATION: Bei Stagnation zufllige Aktionen einfgen
        original_action = action.copy()
        action = self._apply_adaptive_exploration(action)
          # Aktionen interpretieren: [delta_x, delta_y, delta_z, gripper_command]
        position_delta = action[:3] * 0.03  # REDUZIERT: 3cm max. Bewegung fr mehr Kontrolle
        gripper_command = action[3]
        
        # Zustzliche Sicherheit fr Aktionen
        position_delta = np.clip(position_delta, -0.08, 0.08)  # Weitere Beschrnkung
        gripper_command = np.clip(gripper_command, -1.0, 1.0)
        
        # Aktuelle End-Effector Position
        current_end_effector_pos = np.array(p.getLinkState(self.robot_id, self.end_effector_link_index)[0])
        
        # KOLLISIONSVERMEIDUNG: Prfe auf Kollision mit Kiste
        try:
            # Teste neue Position bevor sie angewendet wird
            test_pos = current_end_effector_pos + position_delta
            
            # Kiste Position
            tray_pos = np.array([0.5, 0.0, 0.02])
            tray_size = 0.15  # Kisten-Radius
            
            # Ist der Roboter zu nah an der Kiste? (auer beim Ablegen)
            distance_to_tray = np.linalg.norm(test_pos[:2] - tray_pos[:2])
            cube_pos, _ = p.getBasePositionAndOrientation(self.cube_id)
            is_grasped = self.cube_grasped
            
            # Kollisionsvermeidung: Nicht zu nah an Kiste auer wenn Wrfel gegriffen
            if distance_to_tray < tray_size + 0.05 and not is_grasped:
                # Bewegung weg von der Kiste
                direction_away = (test_pos[:2] - tray_pos[:2])
                if np.linalg.norm(direction_away) > 0:
                    direction_away = direction_away / np.linalg.norm(direction_away)
                    # Korrigiere die XY-Bewegung
                    position_delta[:2] = direction_away * 0.02  # Sanfte Bewegung weg
                    if self.debug_mode:
                        print(" Kollisionsvermeidung: Bewegung von Kiste weg")
        except Exception:
            pass
        
        # Neue Zielposition berechnen
        target_pos = current_end_effector_pos + position_delta
        
        # ERWEITERTE Arbeitsraum-Limits fr bessere Sicherheit
        target_pos[0] = np.clip(target_pos[0], -0.6, 0.8)   # X-Limits (erweitert)
        target_pos[1] = np.clip(target_pos[1], -0.6, 0.6)   # Y-Limits  
        target_pos[2] = np.clip(target_pos[2], 0.08, 0.7)   # Z-Limits (nicht zu tief)
        
        # INVERSE KINEMATIK
        target_orientation = p.getQuaternionFromEuler([0, math.pi, 0])  # Greifer nach unten
        
        try:
            joint_angles = p.calculateInverseKinematics(
                bodyUniqueId=self.robot_id,
                endEffectorLinkIndex=self.end_effector_link_index,
                targetPosition=target_pos,
                targetOrientation=target_orientation,
                maxNumIterations=50,
                residualThreshold=1e-3
            )
        except:
            # Fallback: Aktuelle Gelenk-Positionen beibehalten
            joint_angles = [p.getJointState(self.robot_id, i)[0] for i in range(7)]
        
        # Arm-Joints setzen
        for i in range(min(7, len(joint_angles))):
            p.setJointMotorControl2(
                self.robot_id, i, p.POSITION_CONTROL,
                targetPosition=joint_angles[i], 
                force=150,
                maxVelocity=0.8
            )        #  INTELLIGENTE GREIFER-KONTROLLE MIT AUTO-GREIF-FUNKTION
        current_gripper_pos = p.getJointState(self.robot_id, 9)[0]
        gripper_target = current_gripper_pos
        
        # Prfe, ob Auto-Greif-Conditions erfllt sind
        try:
            end_effector_pos = np.array(p.getLinkState(self.robot_id, self.end_effector_link_index)[0])
            cube_pos, _ = p.getBasePositionAndOrientation(self.cube_id)
            cube_pos = np.array(cube_pos)
            
            # Distanzen berechnen
            distance_to_cube = np.linalg.norm(end_effector_pos - cube_pos)
            xy_distance = np.linalg.norm(end_effector_pos[:2] - cube_pos[:2])
            height_above_cube = end_effector_pos[2] - cube_pos[2]
              #  PRZISES AUTO-GREIF-SYSTEM: Optimale teilweise Schlieung fr perfekten Wrfelgriff
            auto_grasp_conditions = (
                xy_distance < 0.06 and           # Gut ber dem Wrfel (6cm XY-Radius)
                0.02 < height_above_cube < 0.08  # Optimale Greif-Hhe (2-8cm ber Wrfel)
            )
            
            if auto_grasp_conditions and current_gripper_pos > 0.025:
                # Automatische optimale Positionierung - berschreibt die Action!
                # Ziel: 0.020 (optimale Greifposition fr Wrfel) statt voll geschlossen
                optimal_gripper_pos = 0.020
                closing_step = 0.005  # Schrittweise Annherung an optimale Position
                
                # Sanft zur optimalen Position annhern
                if current_gripper_pos > optimal_gripper_pos + closing_step:
                    gripper_target = current_gripper_pos - closing_step
                else:
                    gripper_target = optimal_gripper_pos
                
                if self.debug_mode:
                    print(f" PRZISIONS-GRIFF aktiviert! XY-Dist: {xy_distance:.3f}m, Hhe: {height_above_cube:.3f}m, Ziel: {optimal_gripper_pos:.4f}")
            
            # Normale Action-basierte Kontrolle nur wenn AUTO-GREIF nicht aktiv
            elif not auto_grasp_conditions:
                if gripper_command > 0.3:  # Manuelles Schlieen
                    gripper_target = max(0.008, current_gripper_pos - 0.005)  # Langsam schlieen
                elif gripper_command < -0.3:  # ffnen  
                    gripper_target = min(0.04, current_gripper_pos + 0.008)   # Schneller ffnen
              #  VERBESSERTES PRZISIONS-GREIFEN: Optimale Position bei Kontakt
            contacts_left = p.getContactPoints(bodyA=self.robot_id, bodyB=self.cube_id, linkIndexA=9)
            contacts_right = p.getContactPoints(bodyA=self.robot_id, bodyB=self.cube_id, linkIndexA=10)
            left_contact = len(contacts_left) > 0
            right_contact = len(contacts_right) > 0
            both_contacts = left_contact and right_contact
            has_contact = left_contact or right_contact
            
            if has_contact and gripper_target < current_gripper_pos:
                # Optimale Greifposition basierend auf Kontaktsituation
                if both_contacts:
                    # Bei beidseitigem Kontakt: optimale Greifposition fr Wrfel
                    gripper_target = 0.020  # Perfekte Position fr den Wrfel
                elif left_contact or right_contact:
                    # Bei einseitigem Kontakt: etwas weiter offen lassen fr bessere Zentrierung
                    gripper_target = 0.022  # Leicht offener fr Anpassung
                
                if self.debug_mode:
                    contact_type = "BEIDSEITIGER KONTAKT" if both_contacts else "EINSEITIGER KONTAKT"
                    print(f" {contact_type} - Przisionsgriff bei {gripper_target:.4f}")
                    
        except Exception as e:
            if self.debug_mode:
                print(f" Fehler in Greifer-Logik: {e}")
            # Fallback: Normale Action-basierte Kontrolle
            if gripper_command > 0.3:
                gripper_target = max(0.008, current_gripper_pos - 0.005)
            elif gripper_command < -0.3:
                gripper_target = min(0.04, current_gripper_pos + 0.008)
        
        # Beide Greifer-Finger synchron steuern
        for joint_idx in [9, 10]:
            p.setJointMotorControl2(
                self.robot_id, joint_idx, p.POSITION_CONTROL,
                targetPosition=gripper_target, 
                force=50,
                maxVelocity=0.3
            )
          # Greif-Status aktualisieren
        self._update_grasp_state()
        
        # Physik-Simulation
        for _ in range(5):
            p.stepSimulation()
            if self.gui:
                time.sleep(1./240.)
                
        # Observation und Belohnung
        obs = self._get_obs()
        reward = self.compute_reward(obs['achieved_goal'], obs['desired_goal'], {})
        
        # Reward als float sicherstellen
        reward = float(reward)
        
        # Erfolg prfen
        success = self._is_success(obs['achieved_goal'], obs['desired_goal'])
        self.success = success
        
        # Episode beendet?
        terminated = success  # Erfolg
        truncated = self.episode_step >= self.max_episode_steps  # Zeitlimit erreicht
          # Aktuelle Wrfel position fr Debug
        cube_pos, _ = p.getBasePositionAndOrientation(self.cube_id)
        cube_pos = np.array(cube_pos)
        
        info = {
            'is_success': success,
            'cube_grasped': self.cube_grasped,
            'episode_step': self.episode_step,
            'cube_to_target_distance': np.linalg.norm(cube_pos - self.target_position),  # DEBUG
            'training_phase': self.training_phase,  # Curriculum Info
            'success_rate': getattr(self, 'recent_success_rate', 0.0),  # Aktuelle Erfolgsrate
        }        
        #  CURRICULUM UPDATE bei Episode-Ende
        if terminated or truncated:
            self._update_curriculum_progress(success)
        
        return obs, reward, terminated, truncated, info
    
    def _update_grasp_state(self):
        try:
            # Aktuelle Positionen
            end_effector_pos = np.array(p.getLinkState(self.robot_id, self.end_effector_link_index)[0])
            cube_pos, _ = p.getBasePositionAndOrientation(self.cube_id)
            cube_pos = np.array(cube_pos)
            
            # NaN-Checks fr Positionen
            if np.any(np.isnan(end_effector_pos)) or np.any(np.isinf(end_effector_pos)):
                print(f" NaN/Inf in end_effector_pos: {end_effector_pos}")
                return  # Abort dieser Aktualisierung
                
            if np.any(np.isnan(cube_pos)) or np.any(np.isinf(cube_pos)):
                print(f" NaN/Inf in cube_pos: {cube_pos}")
                return  # Abort dieser Aktualisierung
            
            distance_to_cube = np.linalg.norm(end_effector_pos - cube_pos)
            
            # NaN-Check fr distance_to_cube
            if np.isnan(distance_to_cube) or np.isinf(distance_to_cube):
                print(f" NaN/Inf in distance_to_cube: {distance_to_cube}")
                return  # Abort dieser Aktualisierung
            
            # Aktuelle Greifer-Positionen
            gripper_pos_left = p.getJointState(self.robot_id, 9)[0]
            gripper_pos_right = p.getJointState(self.robot_id, 10)[0]
            
            # NaN-Check fr Greifer-Positionen
            if np.isnan(gripper_pos_left) or np.isnan(gripper_pos_right):
                print(f" NaN in gripper positions: left={gripper_pos_left}, right={gripper_pos_right}")
                return  # Abort dieser Aktualisierung
                
            avg_gripper_pos = (gripper_pos_left + gripper_pos_right) / 2.0
            
        except Exception as e:
            if self.debug_mode:
                print(f" Fehler in _update_grasp_state: {e}")
            return  # Abort bei Fehler
          # PRZISIONS-GREIFEN: Nah am Wrfel UND Greifer teilweise geschlossen UND Kontakt vorhanden
        if distance_to_cube < 0.08 and 0.015 < avg_gripper_pos < 0.025:
            if self.grasp_constraint is None:                # Prüfe Kontakt zwischen Greifer-Fingern und Würfel
                contacts_left = p.getContactPoints(bodyA=self.robot_id, bodyB=self.cube_id, linkIndexA=9)
                contacts_right = p.getContactPoints(bodyA=self.robot_id, bodyB=self.cube_id, linkIndexA=10)
                
                # Berechne XY-Distanz für Kontakt-Prüfung
                end_effector_pos = np.array(p.getLinkState(self.robot_id, self.end_effector_link_index)[0])
                cube_pos, _ = p.getBasePositionAndOrientation(self.cube_id)
                cube_pos = np.array(cube_pos)
                xy_distance = np.linalg.norm(end_effector_pos[:2] - cube_pos[:2])
                
                # Prüfe auf beidseitigen Kontakt für optimalen Griff
                left_contact = len(contacts_left) > 0
                right_contact = len(contacts_right) > 0
                both_contacts = left_contact and right_contact
                if both_contacts or (left_contact or right_contact) and xy_distance < 0.04:
                    # Berechne relative Position des Wrfels zum Greifer
                    end_effector_state = p.getLinkState(self.robot_id, self.end_effector_link_index)
                    end_effector_orn = end_effector_state[1]  # Orientierung
                    
                    # End-Effector-Orientierung fr Berechnung der relativen Position
                    end_effector_matrix = np.reshape(p.getMatrixFromQuaternion(end_effector_orn), (3, 3))
                    
                    # Relativer Vektor vom End-Effector zum Wrfel
                    relative_pos = cube_pos - end_effector_pos
                    
                    # Transformiere in End-Effector-lokales Koordinatensystem
                    local_pos = np.dot(end_effector_matrix.T, relative_pos)
                    
                    # Feinabstimmung der Positionierung fr bessere Zentrierung
                    if both_contacts:
                        local_pos[0] = 0.0  # X-Zentrierung bei beidseitigem Kontakt
                    else:
                        # Bei einseitigem Kontakt: leichte Korrektur zur Mitte
                        correction = 0.01 if left_contact else -0.01
                        local_pos[0] += correction
                    
                    # Hhenanpassung fr optimale Position
                    local_pos[2] -= 0.01  # Leicht nach unten versetzt fr stabilen Griff
                    
                    # VERBESSERTE CONSTRAINT-ERSTELLUNG fr przise Wrfelplatzierung
                    self.grasp_constraint = p.createConstraint(
                        parentBodyUniqueId=self.robot_id,
                        parentLinkIndex=self.end_effector_link_index,
                        childBodyUniqueId=self.cube_id,
                        childLinkIndex=-1,
                        jointType=p.JOINT_FIXED,
                        jointAxis=[0, 0, 0],
                        parentFramePosition=local_pos,  # Verwende lokale Position fr przise Platzierung
                        childFramePosition=[0, 0, 0]    # Wrfelmitte
                    )
                    
                    # Nach erfolgreichem Griff: optimale Greifposition einstellen und halten
                    optimal_gripper_pos = 0.020  # Optimale Position fr Wrfel
                    for joint_idx in [9, 10]:
                        p.setJointMotorControl2(
                            self.robot_id, joint_idx, p.POSITION_CONTROL,
                            targetPosition=optimal_gripper_pos, 
                            force=60,  # Erhhte Kraft fr stabilen Halt
                            maxVelocity=0.3
                        )
                    
                    self.cube_grasped = True
                    if self.debug_mode:
                        grip_type = "PRZISER MITTENGRIFF" if both_contacts else "KORRIGIERTER SEITENGRIFF"
                        print(f" {grip_type}! Distanz: {distance_to_cube:.4f}, Greifer: {avg_gripper_pos:.4f}")

        # LOSLASSEN: Greifer weit geffnet ODER strategisches Loslassen in der Kiste
        elif avg_gripper_pos > 0.025 and self.grasp_constraint is not None:
            # Prfe ob wir ber der Kiste sind fr kontrolliertes Loslassen
            cube_to_target_dist = np.linalg.norm(cube_pos - self.target_position)
            
            if cube_to_target_dist < 0.12:  # Nah genug an der Kiste
                p.removeConstraint(self.grasp_constraint)
                self.grasp_constraint = None
                self.cube_grasped = False
                if self.debug_mode:
                    print(" STRATEGISCHES LOSLASSEN in der Kiste!")
            else:
                # Notfall-Loslassen wenn Greifer zu weit geffnet
                p.removeConstraint(self.grasp_constraint)
                self.grasp_constraint = None
                self.cube_grasped = False
                if self.debug_mode:
                    print(" NOTFALL-LOSLASSEN: Greifer zu weit geffnet")
    
    def compute_reward(self, achieved_goal, desired_goal, info):
        # Sicherstellen, dass achieved_goal und desired_goal numpy arrays sind
        achieved_goal = np.array(achieved_goal, dtype=np.float32)
        desired_goal = np.array(desired_goal, dtype=np.float32)        
        # Batch-Processing fr HER
        if len(achieved_goal.shape) > 1:  # Multiple episodes (batch)
            batch_size = achieved_goal.shape[0]
            rewards = np.zeros(batch_size, dtype=np.float32)
            
            for i in range(batch_size):
                rewards[i] = self._compute_single_reward(achieved_goal[i], desired_goal[i])
            
            return rewards 
        
        else:  # Single episode
            reward = self._compute_single_reward(achieved_goal, desired_goal)
            return np.array([reward], dtype=np.float32) 
    
    def _compute_single_reward(self, achieved_goal, desired_goal):
        
        # Input-Validation mit NaN-Schutz
        achieved_goal = np.array(achieved_goal, dtype=np.float32)
        desired_goal = np.array(desired_goal, dtype=np.float32)
        
        if np.any(np.isnan(achieved_goal)) or np.any(np.isinf(achieved_goal)):
            achieved_goal = np.nan_to_num(achieved_goal, nan=0.0, posinf=1.0, neginf=-1.0)
            
        if np.any(np.isnan(desired_goal)) or np.any(np.isinf(desired_goal)):
            desired_goal = np.nan_to_num(desired_goal, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Grundlegende Werte extrahieren
        cube_pos = achieved_goal[:3]
        is_grasped = achieved_goal[3] > 0.5
        gripper_to_cube_dist = max(0.0, achieved_goal[4])
        target_pos = desired_goal[:3]
        cube_to_target_dist = np.linalg.norm(cube_pos - target_pos)
        
        # NaN-Checks fr berechnete Werte
        if np.isnan(cube_to_target_dist) or np.isinf(cube_to_target_dist):
            cube_to_target_dist = 1.0
        if np.isnan(gripper_to_cube_dist) or np.isinf(gripper_to_cube_dist):
            gripper_to_cube_dist = 1.0
        
        total_reward = 0.0
        phase_info = ""
    
        # PHASE 1: ANNÄHERUNG ZUM WÜRFEL - NUR BEI VERBESSERUNG BELOHNEN
        # Verbesserungsbonus für kontinuierliche Annäherung - NUR bei tatsächlicher Verbesserung!
        if hasattr(self, 'last_gripper_to_cube_dist'):
            if gripper_to_cube_dist < self.last_gripper_to_cube_dist - 0.01:
                improvement = self.last_gripper_to_cube_dist - gripper_to_cube_dist
                # Skaliere Belohnung basierend auf Entfernung - weniger bei näherem Abstand
                if gripper_to_cube_dist > 0.15:
                    total_reward += improvement * 25.0  # Starker Verbesserungsbonus für weit entfernte Annäherung
                elif gripper_to_cube_dist > 0.08:
                    total_reward += improvement * 15.0  # Mittlerer Bonus für mittlere Distanz
                else:
                    total_reward += improvement * 8.0   # Geringerer Bonus für Feinpositionierung
                phase_info += f"IMPROVE({improvement:.3f}) "

        # PHASE 2: POSITIONIERUNG ÜBER DEM WÜRFEL
        try:
            end_effector_pos = np.array(p.getLinkState(self.robot_id, self.end_effector_link_index)[0])
            if np.any(np.isnan(end_effector_pos)):
                end_effector_pos = np.array([0, 0, 0.5])
                  # XY-Alignment: Roboter soll sich über den Würfel bewegen
            xy_distance = np.linalg.norm(end_effector_pos[:2] - cube_pos[:2])
            height_above_cube = end_effector_pos[2] - cube_pos[2]
            
            # KORRIGIERT: XY-Positionierung nur bei VERBESSERUNG belohnen, nicht kontinuierlich!
            if gripper_to_cube_dist < 0.15:  # Nah genug für Positionierung
                # Prüfe auf Verbesserung der XY-Position
                if hasattr(self, 'last_xy_distance'):
                    if xy_distance < self.last_xy_distance - 0.005:  # Signifikante Verbesserung (5mm)
                        xy_improvement = self.last_xy_distance - xy_distance
                        alignment_reward = xy_improvement * 50.0  # Belohnung für XY-Verbesserung
                        total_reward += alignment_reward
                        phase_info += f"XY_IMPROVE({alignment_reward:.1f}) "
                
                # Speichere aktuelle XY-Distanz für nächsten Vergleich
                self.last_xy_distance = xy_distance
                
                # KORRIGIERT: Höhenbelohnung nur bei ABWÄRTSBEWEGUNG
                if hasattr(self, 'last_z_pos'):
                    z_movement = self.last_z_pos - end_effector_pos[2]  # Positiv = Absenkung
                    if z_movement > 0.003 and xy_distance < 0.08:  # Signifikante Absenkung bei guter Position
                        descent_reward = z_movement * 200.0  # Starke Belohnung für tatsächliche Absenkung
                        total_reward += descent_reward
                        phase_info += f"DESCENDING({descent_reward:.1f}) "
                        
                # Aktualisiere Z-Position für nächsten Step
                self.last_z_pos = end_effector_pos[2]
        except Exception:
            xy_distance = 1.0
            height_above_cube = 0.5
        
        # PHASE 3: KONTROLLIERTE ABSENKUNG
        # Wenn gut positioniert, belohne Absenkung zum Wrfel
        try:
            if xy_distance < 0.08 and height_above_cube > 0.03:
                # Progressive Belohnung fr Absenkung
                if 0.03 < height_above_cube < 0.15:
                    descent_progress = (0.15 - height_above_cube) / 0.12
                    total_reward += descent_progress * 15.0  # Sehr hohe Belohnung fr Absenkung
                    phase_info += f"DESCENT({descent_progress:.2f}) "                # REINES RL-TRAINING: Verstrkte Anti-Hovering Belohnungsstruktur
                if not hasattr(self, 'hovering_steps'):
                    self.hovering_steps = 0
                    self.last_z_pos = end_effector_pos[2]
                elif not hasattr(self, 'last_z_pos'):
                    self.last_z_pos = end_effector_pos[2]
                    
                if xy_distance < 0.06 and height_above_cube > 0.10:  # Gut positioniert aber zu hoch
                    self.hovering_steps += 1
                    
                    # Erkennen, ob der Greifer sich nach unten bewegt
                    is_descending = (end_effector_pos[2] < self.last_z_pos - 0.003)  # Mindestens 3mm Absenkung
                    
                    # DEUTLICH VERSTRKTE RL-BESTRAFUNG fr Hovering - muss extrem stark sein
                    # Quadratische Straffunktion fr schnellere Eskalation bei lngerem Hovering
                    hover_steps_penalty = min(self.hovering_steps, 30)  # Cap bei 30 Steps
                    hover_penalty = -2.0 * (hover_steps_penalty * hover_steps_penalty / 15)  # Quadratische Eskalation bis -120
                    total_reward += hover_penalty
                    
                    # MASSIV verstrkte Belohnung für jede Abwrtsbewegung whrend des Hoverings
                    if is_descending:
                        # Höhere Belohnung je lnger das Hovering dauert (verzweifeltere Situation)
                        urgency_factor = min(3.0, 1.0 + self.hovering_steps / 10.0)  # 1.0 bis 3.0
                        descent_motivation = 25.0 * urgency_factor  # Bis zu 75.0 bei langem Hovering!
                        total_reward += descent_motivation
                        phase_info += f"DESCENT_BONUS({descent_motivation:.1f}) "
                    
                    phase_info += f"HOVER_PENALTY({hover_penalty:.1f}) "
                else:
                    self.hovering_steps = 0
                
                # Z-Position fr nchste Vergleiche speichern
                self.last_z_pos = end_effector_pos[2]
            else:
                self.hovering_steps = 0
                if hasattr(self, 'last_z_pos'):
                    self.last_z_pos = end_effector_pos[2]
        except Exception:
            pass        # 
        # PHASE 4: INTELLIGENTES GREIFEN - EVENT-BASIERT
        # Belohnung für das ERREICHEN der optimalen Greif-Position einmalig
        try:
            # Auto-Greif-Bedingungen prüfen
            auto_grasp_zone = (
                xy_distance < 0.06 and           # Gut über dem Würfel (6cm XY-Radius)
                0.02 < height_above_cube < 0.08  # Optimale Greif-Höhe (2-8cm über Würfel)
            )
            
            # KORRIGIERT: Nur belohnen beim ERSTEN MAL in der Auto-Grasp-Zone
            if auto_grasp_zone:
                if not hasattr(self, 'was_in_auto_grasp_zone') or not self.was_in_auto_grasp_zone:
                    auto_grasp_reward = 25.0  # Einmalige Belohnung für das Erreichen der perfekten Position
                    total_reward += auto_grasp_reward
                    phase_info += f"REACHED_GRASP_ZONE({auto_grasp_reward:.0f}) "
                    self.was_in_auto_grasp_zone = True
            else:
                # Reset der Zone, wenn verlassen
                self.was_in_auto_grasp_zone = False
                
        except Exception:
            xy_distance = 1.0
            height_above_cube = 0.5
          # KORRIGIERTE Greiflogik - event-basiert, nicht kontinuierlich
        # Nur bei VERBESSERUNG der Nähe belohnen
        if hasattr(self, 'last_close_distance'):
            if gripper_to_cube_dist < 0.08 and gripper_to_cube_dist < self.last_close_distance - 0.005:
                proximity_improvement = self.last_close_distance - gripper_to_cube_dist
                total_reward += proximity_improvement * 50.0  # Belohnung für das Näher-Kommen
                phase_info += f"GETTING_CLOSER({proximity_improvement:.3f}) "
        
        # Speichere für nächsten Vergleich
        if gripper_to_cube_dist < 0.08:
            self.last_close_distance = gripper_to_cube_dist
            
        # Prüfe auf Kontakt - EVENT-BASIERT
        try:
            contacts_left = p.getContactPoints(bodyA=self.robot_id, bodyB=self.cube_id, linkIndexA=9)
            contacts_right = p.getContactPoints(bodyA=self.robot_id, bodyB=self.cube_id, linkIndexA=10)
            has_contact = len(contacts_left) > 0 or len(contacts_right) > 0
            
            # Nur beim ERSTEN Kontakt belohnen
            if has_contact:
                if not hasattr(self, 'contact_achieved') or not self.contact_achieved:
                    total_reward += 20.0  # Einmalige starke Belohnung für ersten Kontakt
                    phase_info += "FIRST_TOUCH! "
                    self.contact_achieved = True
            else:
                self.contact_achieved = False  # Reset wenn Kontakt verloren
        except Exception:
            pass
        
        # Mächtige Belohnung fr erfolgreiches Greifen
        if is_grasped:
            total_reward += 35.0  # Erhht von 30.0 fr strkere Motivation
            phase_info += "GRASPED! "
            phase_info += "GRASPED! "
        
    
        # PHASE 5: TRANSPORT ZUM ZIEL
        if is_grasped:
            # Progressive Belohnung fr Transport
            transport_progress = max(0, (1.0 - cube_to_target_dist) / 1.0)
            total_reward += transport_progress * 20.0
            phase_info += f"TRANSPORT({transport_progress:.2f}) "
            
            # Verbesserungsbonus fr Transport
            if hasattr(self, 'last_cube_to_target_dist'):
                if cube_to_target_dist < self.last_cube_to_target_dist - 0.01:
                    transport_improvement = self.last_cube_to_target_dist - cube_to_target_dist
                    total_reward += transport_improvement * 30.0
                    phase_info += f"PROGRESS({transport_improvement:.3f}) "
            
            # Bonus fr Erreichen des Ziels
            if cube_to_target_dist < self.distance_threshold:
                total_reward += 60.0  # Massive Erfolgsbelohnung
                phase_info += "SUCCESS! "
        
        # 
        # KOLLISIONSVERMEIDUNG
        # 
        # Einfache Strafe fr Kollision mit der Kiste (auer beim Ablegen)
        try:
            if not is_grasped:  # Nur wenn kein Wrfel gegriffen ist
                tray_contacts = p.getContactPoints(bodyA=self.robot_id, bodyB=self.tray_id)
                if len(tray_contacts) > 0:
                    total_reward -= 3.0  # Strafe fr Kollision
                    phase_info += "COLLISION! "
        except Exception:
            pass
        
        # Tracking-Werte aktualisieren
        self.last_gripper_to_cube_dist = gripper_to_cube_dist
        self.last_cube_to_target_dist = cube_to_target_dist
        
        # Debug-Ausgabe (reduziert)
        if self.debug_mode and self.episode_step % 100 == 0:
            print(f"Step {self.episode_step:3d}: Reward={total_reward:+6.2f} | {phase_info}")
            print(f"  DistCube: {gripper_to_cube_dist:.3f}m | XY: {xy_distance:.3f}m | Height: {height_above_cube:.3f}m")
        
        # Reward begrenzen und zurckgeben
        total_reward = np.clip(total_reward, -10.0, 100.0)
        return float(total_reward)
        
    def _is_success(self, achieved_goal, desired_goal):
        # Sicherstellen, dass Inputs numpy arrays sind
        achieved_goal = np.array(achieved_goal)
        desired_goal = np.array(desired_goal)
        
        if len(achieved_goal.shape) == 1:  # Einzelne Episode
            # achieved_goal = [cube_x, cube_y, cube_z, is_grasped, gripper_to_cube_dist]
            cube_pos = achieved_goal[:3]
            is_grasped = achieved_goal[3] > 0.5
            
            # desired_goal = [target_x, target_y, target_z, should_be_grasped, optimal_gripper_dist]  
            target_pos = desired_goal[:3]
            
            # Erfolg = Wrfel wurde erfolgreich zur Kiste transportiert
            cube_to_target_distance = np.linalg.norm(cube_pos - target_pos)
            success = is_grasped and (cube_to_target_distance < self.distance_threshold)
            
            return success
        else:
            # Batch-Processing
            batch_size = achieved_goal.shape[0]
            successes = np.zeros(batch_size, dtype=bool)
            
            for i in range(batch_size):
                cube_pos = achieved_goal[i, :3]
                is_grasped = achieved_goal[i, 3] > 0.5
                target_pos = desired_goal[i, :3]
                cube_to_target_distance = np.linalg.norm(cube_pos - target_pos)
                successes[i] = is_grasped and (cube_to_target_distance < self.distance_threshold)
            
            return successes
            
    def _get_adaptive_cube_position(self):
        # Tray-Parameter (entspricht der tray_size Definition im Code)
        tray_center_x = 0.5
        tray_center_y = 0.0
        tray_size = 0.15  # Radius der Tray-Box
        
        # Gespiegelte Position: Tray bei +0.5, Wrfel-Spawn-Bereich bei -0.5
        spawn_center_x = -tray_center_x  # -0.5 (gespiegelt)
        spawn_center_y = tray_center_y   # 0.0 (gleiche Y-Position)
        
        # Wrfel spawnt in exakt der gleichen Flche wie die Tray-Box, nur gespiegelt
        cube_x = np.random.uniform(spawn_center_x - tray_size, spawn_center_x + tray_size)  # [-0.65, -0.35]
        cube_y = np.random.uniform(spawn_center_y - tray_size, spawn_center_y + tray_size)  # [-0.15, +0.15]
        
        if self.debug_mode:
            print(f" Wrfel spawnt bei: [{cube_x:.3f}, {cube_y:.3f}] - Gespiegelte Tray-Flche")
        
        return cube_x, cube_y
    
    def _apply_adaptive_exploration(self, action):
        # Aktuelle Metriken berechnen
        try:
            end_effector_pos = np.array(p.getLinkState(self.robot_id, self.end_effector_link_index)[0])
            cube_pos, _ = p.getBasePositionAndOrientation(self.cube_id)
            cube_pos = np.array(cube_pos)
            
            gripper_to_cube_dist = np.linalg.norm(end_effector_pos - cube_pos)
            cube_to_target_dist = np.linalg.norm(cube_pos - self.target_position)
              #  HOVERING DETECTION: Greifer ber dem Wrfel aber nicht absenkend
            horizontal_dist = np.sqrt((end_effector_pos[0] - cube_pos[0])**2 + (end_effector_pos[1] - cube_pos[1])**2)
            height_diff = end_effector_pos[2] - cube_pos[2]
            
            # VERBESSERTE HOVERING-ERKENNUNG: Nur bei sehr prziser Positionierung
            # Reduziert von 12cm auf 6cm fr bessere Feinpositionierung
            is_hovering = (horizontal_dist < 0.06 and height_diff > 0.06 and height_diff < 0.25)
            
        except Exception:
            # Bei Fehlern: keine Modifikation
            return action
        
        #  STAGNATION DETECTION
        improved = False
        
        # Verbesserte Annherung an Wrfel?
        if gripper_to_cube_dist < self.stagnation_detection['best_distance_to_cube'] - 0.01:
            self.stagnation_detection['best_distance_to_cube'] = gripper_to_cube_dist
            improved = True
            
        # Verbesserte Wrfel-Position?
        if self.cube_grasped and cube_to_target_dist < self.stagnation_detection['best_cube_to_target'] - 0.01:
            self.stagnation_detection['best_cube_to_target'] = cube_to_target_dist
            improved = True
            
        # Greif-Versuch erkannt?
        gripper_command = action[3]
        if abs(gripper_command) > 0.3:  # Aktive Greifer-Aktion
            self.stagnation_detection['steps_without_grasp_attempt'] = 0
            improved = True
        else:
            self.stagnation_detection['steps_without_grasp_attempt'] += 1
        
        #  HOVERING DETECTION UPDATE
        if is_hovering:
            self.stagnation_detection['hovering_steps'] += 1
            if self.debug_mode and self.stagnation_detection['hovering_steps'] % 10 == 0:
                print(f" HOVERING DETECTED: {self.stagnation_detection['hovering_steps']} steps ber dem Wrfel")
        else:
            self.stagnation_detection['hovering_steps'] = 0
        
        # Update Stagnation Counter
        if improved:
            self.stagnation_detection['no_improvement_steps'] = 0
            self.stagnation_detection['exploration_boost_active'] = False
            if self.debug_mode and self.episode_step % 50 == 0:
                print(f" Verbesserung erkannt - Exploration zurckgesetzt")
        else:
            self.stagnation_detection['no_improvement_steps'] += 1
        
        #  EXPLORATION BOOST AKTIVIEREN
        exploration_threshold = 30  # Steps ohne Verbesserung
        grasp_threshold = 50        # Steps ohne Greif-Versuch
        hovering_threshold = self.stagnation_detection['hovering_threshold']  # Steps ber Wrfel
        
        force_exploration = (
            self.stagnation_detection['no_improvement_steps'] > exploration_threshold or
            self.stagnation_detection['steps_without_grasp_attempt'] > grasp_threshold or
            self.stagnation_detection['hovering_steps'] > hovering_threshold  # NEU: Hovering-Erkennung
        )        #  VOLLSTNDIG RL-BASIERTE ANTI-HOVERING STRATEGIE
        # Keine Aktionsintervention mehr - Agent muss komplett durch Reward-Signal lernen
        if self.stagnation_detection['hovering_steps'] > hovering_threshold:
            if self.debug_mode:
                print(f" REINES RL-TRAINING FR ANTI-HOVERING! {self.stagnation_detection['hovering_steps']} steps hovering")
                
            # Nur Information speichern fr Debug/Monitoring - keine Aktionsnderung!
            self.stagnation_detection['descent_motivation_active'] = True
        
        if force_exploration:
            if not self.stagnation_detection['exploration_boost_active']:
                self.stagnation_detection['exploration_boost_active'] = True
                self.stagnation_detection['forced_action_counter'] = 0
                if self.debug_mode:
                    print(f" EXPLORATION BOOST aktiviert! Stagnation seit {self.stagnation_detection['no_improvement_steps']} Steps")
            
            #  VERSCHIEDENE EXPLORATION STRATEGIEN
            strategy = self.stagnation_detection['forced_action_counter'] % 5  # ERWEITERT von 4 auf 5
            
            if strategy == 0:  # RANDOM MOVEMENT
                exploration_noise = np.random.normal(0, 0.5, 3)
                action[:3] = np.clip(action[:3] + exploration_noise, -1.0, 1.0)
                if self.debug_mode and self.episode_step % 20 == 0:
                    print(f" Random Movement Exploration")
                    
            elif strategy == 1:  # FORCE GRIPPER ACTION
                if gripper_to_cube_dist < 0.15:  # Nah genug zum Greifen
                    action[3] = 0.8 if np.random.random() > 0.5 else -0.8  # Schlieen oder ffnen
                    if self.debug_mode and self.episode_step % 20 == 0:
                        print(f" Forced Gripper Action")
                else:
                    # Bewege zum Wrfel
                    direction = (cube_pos - end_effector_pos)
                    direction = direction / (np.linalg.norm(direction) + 1e-6)
                    action[:3] = direction * 0.8
                    
            elif strategy == 2:  # VERTICAL EXPLORATION
                action[2] = np.random.uniform(-0.8, 0.8)  # Zufllige Z-Bewegung
                if self.debug_mode and self.episode_step % 20 == 0:
                    print(f" Vertical Exploration")
                    
            elif strategy == 3:  # RESET TO CUBE
                # Direkte Bewegung zum Wrfel
                if gripper_to_cube_dist > 0.05:
                    direction = (cube_pos - end_effector_pos)
                    direction = direction / (np.linalg.norm(direction) + 1e-6)
                    action[:3] = direction * 0.7
                    action[3] = 0.0  # Greifer neutral
                    if self.debug_mode and self.episode_step % 20 == 0:
                        print(f" Direct Cube Approach")
                        
            else:  # NEU: FORCED DESCENT STRATEGY
                # Spezielle Strategie fr Absenkung wenn nah am Wrfel
                if gripper_to_cube_dist < 0.15:
                    action[2] = -0.7  # Starke Absenkung
                    action[0] *= 0.2  # Minimale horizontale Bewegung
                    action[1] *= 0.2
                    if self.debug_mode and self.episode_step % 20 == 0:
                        print(f" Forced Descent Strategy")
            
            self.stagnation_detection['forced_action_counter'] += 1
            
            # Exploration nach einiger Zeit zurcksetzen
            if self.stagnation_detection['forced_action_counter'] > 25:  # ERHHT von 20
                self.stagnation_detection['no_improvement_steps'] = 0
                self.stagnation_detection['exploration_boost_active'] = False
                if self.debug_mode:
                    print(f" Exploration Boost zurckgesetzt nach 25 Steps")
        else:
            # Reset Descent Motivation wenn keine Exploration aktiv
            self.stagnation_detection['descent_motivation_active'] = False
        
        return action
    
    def _update_main_goal_focus(self, success):
        if success:
            self.episodes_since_success = 0
            self.urgency_factor = 1.0
            if self.debug_mode:
                print(f" Erfolg! Urgency zurckgesetzt")
        else:
            self.episodes_since_success += 1
            # Erhhe Dringlichkeit graduell
            self.urgency_factor = min(3.0, 1.0 + (self.episodes_since_success * 0.1))
            if self.debug_mode and self.episodes_since_success % 10 == 0:
                print(f" {self.episodes_since_success} Episoden ohne Erfolg - Urgency: {self.urgency_factor:.2f}")
        
        # Reduziere Unterziel-Belohnungen ber Zeit
        self.sub_goal_decay = max(0.5, self.sub_goal_decay * 0.999)
        
        # Erhhe Hauptziel-Gewichtung bei wiederholten Fehlschlgen
        self.main_goal_weight = min(5.0, 1.0 + (self.episodes_since_success * 0.05))
    
    def _update_curriculum_progress(self, success):
        if success:
            self.success_streak += 1
        else:
            self.success_streak = max(0, self.success_streak - 1)
        
        # Berechne gleitende Erfolgsrate (letzte 20 Episodes)
        if not hasattr(self, 'recent_successes'):
            self.recent_successes = []
            
        self.recent_successes.append(success)
        if len(self.recent_successes) > 20:
            self.recent_successes.pop(0)  # Entferne lteste
        
        self.recent_success_rate = np.mean(self.recent_successes)
        
        # Phase-Upgrade prfen
        if self.training_phase < 3:  # Nicht ber Expert-Level hinaus
            current_threshold = self.phase_success_threshold[self.training_phase]
            
            if (self.recent_success_rate > current_threshold and 
                self.success_streak >= 5 and 
                len(self.recent_successes) >= 10):  # Mindestens 10 Episodes fr valide Rate
                
                self.training_phase += 1
                self.success_streak = 0
                if self.debug_mode:
                    print(f" CURRICULUM UPGRADE! Phase {self.training_phase-1}  {self.training_phase}")
                    print(f"   Erfolgsrate: {self.recent_success_rate:.1%}, Streak: {self.success_streak}")


class SAC_HER_TrainingCallback(BaseCallback):
    
    def __init__(self, save_freq=25000, save_path='./sac_her_models/', verbose=1):
        super(SAC_HER_TrainingCallback, self).__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        
        # Tracking Listen
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_successes = []
        
        self.current_episode_reward = 0
        self.current_episode_length = 0
        
        os.makedirs(save_path, exist_ok=True)
    
    def _on_step(self) -> bool:
        self.current_episode_reward += self.locals['rewards'][0]
        self.current_episode_length += 1
        
        # DEBUG: Verfolge minimale Distanz zur Kiste pro Episode
        if 'infos' in self.locals and len(self.locals['infos']) > 0:
            info = self.locals['infos'][0]
            if 'cube_to_target_distance' in info:
                if not hasattr(self, 'min_distance_this_episode'):
                    self.min_distance_this_episode = float('inf')
                self.min_distance_this_episode = min(self.min_distance_this_episode, info['cube_to_target_distance'])
        
        # Episode beendet?
        if 'dones' in self.locals and self.locals['dones'][0]:
            # Episode beendet
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_lengths.append(self.current_episode_length)
              # Success tracking
            if 'infos' in self.locals and len(self.locals['infos']) > 0:
                success = self.locals['infos'][0].get('is_success', False)
                self.episode_successes.append(success)
                
                #  CURRICULUM LEARNING TRACKING
                training_phase = self.locals['infos'][0].get('training_phase', 0)
                success_rate = self.locals['infos'][0].get('success_rate', 0.0)
                
                # Speichere Curriculum-Metriken
                if not hasattr(self, 'training_phases'):
                    self.training_phases = []
                    self.success_rates = []
                self.training_phases.append(training_phase)
                self.success_rates.append(success_rate)
                
                # DEBUG: Zeige minimale Distanz
                if hasattr(self, 'min_distance_this_episode'):
                    print(f"Episode {len(self.episode_rewards)}: Reward={self.current_episode_reward:.0f}, "
                          f"Success={success}, Phase={training_phase}, Rate={success_rate:.2%}, "
                          f"Min Distance={self.min_distance_this_episode:.3f}m")
                    self.min_distance_this_episode = float('inf')
            
            # Stats ausgeben
           
            if len(self.episode_rewards) % 10 == 0:
                recent_rewards = self.episode_rewards[-10:]
                recent_successes = self.episode_successes[-10:]
                
                avg_reward = np.mean(recent_rewards)
                success_rate = np.mean(recent_successes) * 100
                
                print(f"Episode {len(self.episode_rewards)}: "
                      f"Avg Reward: {avg_reward:.1f}, "
                      f"Success Rate: {success_rate:.1f}%")
            
            # Reset fr nchste Episode
            self.current_episode_reward = 0
            self.current_episode_length = 0
        
        # Model speichern
        if self.n_calls % self.save_freq == 0:
            model_path = os.path.join(self.save_path, f'sac_her_model_step_{self.n_calls}.zip')
            self.model.save(model_path)
            print(f"SAC+HER Model gespeichert: {model_path}")
            
            # Plot erstellen            # Plot erstellen
            self._create_plot()
            
        return True
    
    def _create_plot(self):
        if len(self.episode_rewards) < 5:
            return
            
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Episode Rewards
        ax1.plot(self.episode_rewards, 'b-', alpha=0.7)
        ax1.set_title('Episode Rewards (SAC+HER)')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Reward')
        ax1.grid(True, alpha=0.3)
        
        # Success Rate (gleitender Durchschnitt)
        if len(self.episode_successes) >= 10:
            window_size = min(20, len(self.episode_successes) // 2)
            success_rates = []
            for i in range(window_size, len(self.episode_successes)):
                rate = np.mean(self.episode_successes[i-window_size:i]) * 100
                success_rates.append(rate)
            
            ax2.plot(range(window_size, len(self.episode_successes)), success_rates, 'g-', linewidth=2)
            ax2.set_title(f'Success Rate (Moving Avg, Window={window_size})')
            ax2.set_xlabel('Episode')
            ax2.set_ylabel('Success Rate (%)')
            ax2.grid(True, alpha=0.3)
            ax2.set_ylim(0, 100)
        
        # Episode Lengths
        ax3.plot(self.episode_lengths, 'purple', alpha=0.7)
        ax3.set_title('Episode Lengths')
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Steps')
        ax3.grid(True, alpha=0.3)
          #  Curriculum Learning Progress (ax4)
        if hasattr(self, 'training_phases') and len(self.training_phases) > 0:
            ax4.plot(self.training_phases, 'orange', linewidth=2, marker='o', markersize=3)
            ax4.set_title(' Curriculum Learning Progress')
            ax4.set_xlabel('Episode')
            ax4.set_ylabel('Training Phase')
            ax4.set_ylim(-0.5, 3.5)
            ax4.set_yticks([0, 1, 2, 3])
            ax4.set_yticklabels(['Easy', 'Medium', 'Hard', 'Expert'])
            ax4.grid(True, alpha=0.3)
        else:
            # Fallback: Replay Buffer Info
            if hasattr(self.model, 'replay_buffer') and hasattr(self.model.replay_buffer, 'size'):
                buffer_sizes = [self.model.replay_buffer.size()] * len(self.episode_rewards)
                ax4.plot(buffer_sizes, 'orange', linewidth=2)
                ax4.set_title('HER Replay Buffer Size')
                ax4.set_xlabel('Episode')
                ax4.set_ylabel('Buffer Size')
                ax4.grid(True, alpha=0.3)
        
        plt.suptitle(f'SAC+HER Training Progress - Step {self.n_calls:,}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        plot_path = os.path.join(self.save_path, f'sac_her_training_plot_{self.n_calls}.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Plot gespeichert: {plot_path}")


def train_sac_her_model(gui=True, total_timesteps=1000000, pretrained_model_path=None):
    print(" Starte SAC + HER Training fr Pick-and-Place...")
    print(" NEW FEATURES:")
    print("   SAC: Off-Policy Algorithmus mit kontinuierlichen Aktionen")
    print("   HER: Hindsight Experience Replay fr sparse rewards")
    print("   Goal-Environment: Automatische HER-Integration")
    print("   Sample-effizientes Lernen aus Fehlschlgen")
    print("   Keine lokalen Optima mehr durch Off-Policy Learning")
    print(f"   Trainingszeit: {total_timesteps:,} Steps")
    
    if pretrained_model_path:
        print(f"   WEITERTRAINING von: {pretrained_model_path}")
    else:
        print("   NEUES TRAINING von Grund auf")
    print()
    
    # Environment erstellen
    env = RobotArmPickPlaceEnvHER(gui=gui, debug_mode=True)
    
    # Fr SAC muss die Environment in DummyVecEnv gewrappt werden
    env = DummyVecEnv([lambda: env])
    
    # SAC mit HER Replay Buffer - ANTI-LOKALE-OPTIMA Optimierung
    if pretrained_model_path and os.path.exists(pretrained_model_path):
        print(f" Lade vortrainiertes Modell: {pretrained_model_path}")
        try:
            model = SAC.load(pretrained_model_path, env=env)
            print(" Modell erfolgreich geladen - setze Training fort!")
            
            # Buffer-Info anzeigen
            if hasattr(model.replay_buffer, 'size'):
                buffer_size = model.replay_buffer.size()
                print(f" Replay Buffer Gre: {buffer_size:,} Erfahrungen")
            
        except Exception as e:
            print(f" Fehler beim Laden des Modells: {e}")
            print(" Starte neues Training von Grund auf...")
            pretrained_model_path = None
    
    if not pretrained_model_path:
        model = SAC(
            "MultiInputPolicy",  # Fr Goal-basierte Environments
            env, 
            replay_buffer_class=HerReplayBuffer,
            replay_buffer_kwargs=dict(
                n_sampled_goal=4,  # Anzahl der HER-Goals pro echtem Goal
                goal_selection_strategy='future',  # Nimm zuknftige Positionen als Goals
            ),
            verbose=1,
            learning_rate=3e-4,      # Standard Learning Rate fr stabile Exploration
            buffer_size=1000000,     # Groer Replay Buffer fr diverse Erfahrungen
            learning_starts=10000,   # MEHR Exploration vor Training-Start
            batch_size=128,          # Kleinere Batches fr stabilere Updates
            tau=0.005,               # Langsame Target Updates
            gamma=0.99,              # HHER fr langfristige Belohnungen
            train_freq=4,            # Trainiere alle 4 Steps (weniger frequent)
            gradient_steps=1,        
            ent_coef='auto',         # KRITISCH: Automatische Entropie fr Exploration
            target_update_interval=1, 
            tensorboard_log="./sac_her_tensorboard/",
            policy_kwargs=dict(
                net_arch=[512, 512, 256],  # Groes Netzwerk fr komplexe Policies
                activation_fn=torch.nn.ReLU,
            ),
            # ANTI-LOKALE-OPTIMA: Erhhte initiale Entropie
            target_entropy='auto',    # Automatische Entropie-Ziel-Anpassung
        )
    
    # Callback
    callback = SAC_HER_TrainingCallback(save_freq=25000, save_path='./sac_her_models/')
      # Training mit Error Handling
    print(f"SAC + HER Training bis zu {total_timesteps:,} Steps...")
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            tb_log_name=f"SAC_HER_PickPlace_{'GUI' if gui else 'NoGUI'}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
    except Exception as e:
        print(f" Training unterbrochen: {e}")
        print(" Speichere aktuellen Modellstand...")
        
        # Notfall-Speicherung
        emergency_path = f"./sac_her_models/emergency_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
        model.save(emergency_path)
        print(f" Notfall-Model gespeichert: {emergency_path}")
        
        # Environment sicher schlieen
        env.envs[0].close()
        
        raise e
    
    # Finales Model speichern
    try:
        final_model_path = f"./sac_her_models/final_sac_her_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
        model.save(final_model_path)
        print(f" Finales Model gespeichert: {final_model_path}")
        
        # Environment sicher schlieen
        env.envs[0].close()
        print(" Training erfolgreich abgeschlossen!")
        
    except Exception as e:
        print(f" Fehler beim finalen Speichern: {e}")
        # Trotzdem versuchen zu schlieen
        try:
            env.envs[0].close()
        except:
            pass
    
    return model, final_model_path


def test_sac_her_model(model_path, gui=True, episodes=10):
    print(f" Teste SAC+HER Model: {model_path}")
    
    env = RobotArmPickPlaceEnvHER(gui=gui, debug_mode=True)
    # Lade das Modell mit der Environment, um HER-Kompatibilitt sicherzustellen
    model = SAC.load(model_path, env=env)
    success_count = 0

    for episode in range(episodes):
        obs, info = env.reset()
        episode_reward = 0
        terminated = False
        truncated = False
        step_count = 0
        
        print(f"\n Episode {episode + 1}/{episodes}")
        
        while not (terminated or truncated) and step_count < 2000:
            # SAC braucht nur die 'observation' Teil
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            # HER Environment gibt reward als array zurck, wir brauchen den Skalar
            reward_scalar = reward[0] if isinstance(reward, np.ndarray) else reward
            episode_reward += reward_scalar
            step_count += 1
            
            if step_count % 300 == 0:
                cube_pos = obs['achieved_goal']
                target_pos = obs['desired_goal']
                distance = np.linalg.norm(cube_pos - target_pos)
                print(f"  Step {step_count}: Distance to goal: {distance:.3f}m")
                
        if info['is_success']:
            success_count += 1
            print(f"   ERFOLG in {step_count} Steps!")
        else:
            print(f"   Fehlgeschlagen nach {step_count} Steps")
            
        print(f"   Episode Reward: {episode_reward:.2f}")
    
    success_rate = success_count / episodes * 100
    
    print(f"\n SAC+HER ENDERGEBNIS:")
    print(f"  Erfolgsrate: {success_count}/{episodes} ({success_rate:.1f}%)")
    
    env.close()


def load_and_test_model(model_path, episodes=5, gui=True):
    print(f" Lade und teste Model: {model_path}")
    
    # IDENTISCHE Environment wie beim Training erstellen
    env = RobotArmPickPlaceEnvHER(gui=gui, debug_mode=True)
    env = DummyVecEnv([lambda: env])
    
    try:
        # Model mit Environment-Referenz laden (fr HER erforderlich)
        model = SAC.load(model_path, env=env)
        print(" SAC+HER Model erfolgreich geladen!")
        
        success_count = 0
        for episode in range(episodes):
            obs = env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action)
                episode_reward += reward[0]
                
                if done[0]:
                    is_success = info[0].get('is_success', False)
                    if is_success:
                        success_count += 1
                    print(f"Episode {episode+1}: Reward={episode_reward:.0f}, Success={'' if is_success else ''}")
                    break
        
        print(f" Erfolgsrate: {success_count}/{episodes} ({success_count/episodes*100:.1f}%)")
        env.envs[0].close()
        return success_count / episodes
        
    except Exception as e:
        print(f" Fehler beim Laden: {e}")
        env.envs[0].close()
        return None


def main():
    print("=== SAC + HER Pick-and-Place (Off-Policy + Hindsight Experience Replay) ===")
    print(" NEW FEATURES:")
    print("    SAC: Kontinuierliche Off-Policy Kontrolle")
    print("    HER: Lernt aus Fehlschlgen durch Goal-Relabeling")
    print("    Goal-Environment: Automatische HER-Integration")
    print("    Sample Efficiency: Viel effizienter als On-Policy")
    print("    Sparse Rewards: Perfekt fr Pick-and-Place Tasks")
    print("    Weitertraining: Bestehende Modelle verbessern")
    print("    Anti-Hovering: Lst das Stagnations-Problem!")
    print("")
    print("1. Neues SAC+HER Training mit GUI")
    print("2. Neues SAC+HER Training ohne GUI") 
    print("3.  Weitertraining eines bestehenden Modells")
    print("4. SAC+HER Modell testen")
    print("5. Environment-Test")
    print("6.  DEBUG: Anti-Hovering System Test")
    print("7. Beenden")
    
    while True:
        choice = input("\nOption whlen (1-7): ").strip()
        
        if choice == '1':
            print(" Starte NEUES SAC+HER Training mit GUI...")
            train_sac_her_model(gui=True, total_timesteps=500000)  # Weniger Steps fr ersten Test
            
        elif choice == '2':
            print(" Starte NEUES SAC+HER Training ohne GUI...")
            train_sac_her_model(gui=False, total_timesteps=10000000)  # Mehr Steps fr Volltraining
            
        elif choice == '3':
            continue_training_existing_model()
            
        elif choice == '4':
            model_files = []
            if os.path.exists('./sac_her_models'):
                model_files = [f for f in os.listdir('./sac_her_models') if f.endswith('.zip')]
            
            if not model_files:
                print(" Keine SAC+HER Modell-Dateien gefunden!")
                continue                
            print("\n Verfgbare SAC+HER Modelle:")
            for i, f in enumerate(model_files, 1):
                print(f"  {i}. {f}")
                
            try:
                idx = int(input("Modell whlen: ")) - 1
                model_filename = model_files[idx]
                model_path = os.path.join('./sac_her_models', model_filename)
                episodes = int(input("Anzahl Test-Episoden (default 5): ") or "5")
                test_sac_her_model(model_path, gui=True, episodes=episodes)
            except (ValueError, IndexError):
                print(" Ungltige Eingabe!")
            
        elif choice == '5':
            print(" SAC+HER Environment-Test...")
            env = RobotArmPickPlaceEnvHER(gui=True, debug_mode=True)
            print("Teste 3 Reset-Zyklen...")
            for i in range(3):
                print(f"\nReset {i+1}:")
                obs, info = env.reset()
                print(f"  Observation Keys: {obs.keys()}")
                print(f"  Observation Shape: {obs['observation'].shape}")
                print(f"  Achieved Goal: {obs['achieved_goal']}")
                print(f"  Desired Goal: {obs['desired_goal']}")
                
                # Paar zufllige Aktionen
                for j in range(10):
                    action = env.action_space.sample() * 0.3  # Kleinere Aktionen
                    obs, reward, terminated, truncated, info = env.step(action)
                    print(f"    Step {j+1}: Reward={reward:.3f}, Terminated={terminated}, Truncated={truncated}, Success={info['is_success']}")
                    if terminated or truncated:
                        break
                    env.close()
            print(" SAC+HER Environment-Test abgeschlossen!")
            
        elif choice == '6':
            print(" Starte Anti-Hovering System Debug...")
            debug_hovering_problem()
            
        elif choice == '7':
            print(" Auf Wiedersehen!")
            break
            
        else:
            print(" Ungltige Option!")


def continue_training_existing_model():
    print(" WEITERTRAINING eines bereits trainierten SAC+HER Modells")
    print("=" * 60)
    
    # Verfgbare Modelle auflisten
    model_files = []
    if os.path.exists('./sac_her_models'):
        model_files = [f for f in os.listdir('./sac_her_models') if f.endswith('.zip')]
    
    if not model_files:
        print(" Keine SAC+HER Modell-Dateien im Ordner './sac_her_models' gefunden!")
        print("   Stellen Sie sicher, dass bereits trainierte Modelle vorhanden sind.")
        return
    
    print(f"\n Verfgbare Modelle ({len(model_files)} gefunden):")
    for i, f in enumerate(model_files, 1):
        # Dateigre anzeigen
        file_path = os.path.join('./sac_her_models', f)
        file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
        print(f"  {i}. {f} ({file_size:.1f} MB)")
    
    try:
        print("\n" + "=" * 60)
        model_idx = int(input(" Welches Modell soll weitertrainiert werden? (Nummer): ")) - 1
        
        if model_idx < 0 or model_idx >= len(model_files):
            print(" Ungltige Modell-Nummer!")
            return
        
        selected_model = model_files[model_idx]
        model_path = os.path.join('./sac_her_models', selected_model)
        
        # Training-Parameter abfragen
        print("\n TRAINING-KONFIGURATION:")
        gui_choice = input(" Mit GUI trainieren? (j/n, default=j): ").strip().lower()
        use_gui = gui_choice != 'n'
        
        # Anzahl Steps
        print("\n Trainings-Umfang:")
        print("  1. Kurz    : 250,000 Steps (~30-60 Min)")
        print("  2. Mittel  : 500,000 Steps (~1-2 Stunden)")
        print("  3. Lang    : 1,000,000 Steps (~2-4 Stunden)")
        print("  4. Sehr Lang: 2,000,000 Steps (~4-8 Stunden)")
        print("  5. Benutzerdefiniert")
        
        duration_choice = input("Trainings-Umfang whlen (1-5, default=2): ").strip()
        
        duration_map = {
            '1': 250000,
            '2': 500000,
            '3': 1000000,
            '4': 2000000,
            '': 500000  # Default
        }
        
        if duration_choice in duration_map:
            total_timesteps = duration_map[duration_choice]
        elif duration_choice == '5':
            try:
                total_timesteps = int(input("Anzahl Trainings-Steps eingeben: "))
                if total_timesteps <= 0:
                    print(" Ungltige Anzahl Steps!")
                    return
            except ValueError:
                print(" Ungltige Eingabe!")
                return
        else:
            total_timesteps = 500000  # Default
        
        print(f"\n STARTE WEITERTRAINING:")
        print(f"  Modell: {selected_model}")
        print(f"  GUI: {'Ja' if use_gui else 'Nein'}")
        print(f"  Trainings-Steps: {total_timesteps:,}")
        
        # Environment erstellen
        env = RobotArmPickPlaceEnvHER(gui=use_gui, debug_mode=True)
        env = DummyVecEnv([lambda: env])
        
        # Vorhandenes Modell laden
        model = SAC.load(model_path, env=env)
        
        # Callback
        callback = SAC_HER_TrainingCallback(save_freq=25000, save_path='./sac_her_models/')
        
        # Training starten
        model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            tb_log_name=f"SAC_HER_PickPlace_{'GUI' if use_gui else 'NoGUI'}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        
        # Finales Model speichern
        final_model_path = f"./sac_her_models/final_sac_her_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
        model.save(final_model_path)
        print(f" Finales Model gespeichert: {final_model_path}")
        
        # Environment schlieen
        env.envs[0].close()
        print(" Weitertraining erfolgreich abgeschlossen!")
        
    except Exception as e:        
        print(f" Fehler beim Weitertraining: {e}")


def debug_hovering_problem():
    print(" DEBUG: ANTI-HOVERING SYSTEM TEST")
    print("=" * 60)
    
    # Environment mit Debug-Modus erstellen
    env = RobotArmPickPlaceEnvHER(gui=True, debug_mode=True)
    
    obs, info = env.reset()
    print(f" Environment initialisiert")
    
    # Test-Sequenz: Roboter ber Wrfel positionieren
    print("\n PHASE 1: Roboter ber Wrfel positionieren")
    
    cube_pos = obs['achieved_goal'][:3]
    print(f" Wrfel-Position: [{cube_pos[0]:.3f}, {cube_pos[1]:.3f}, {cube_pos[2]:.3f}]")
    
    # Bewegung ber den Wrfel (horizontal)
    for step in range(50):
        # Bewegung zum Wrfel (horizontal)
        action = np.array([
            0.3 if step < 20 else 0.0,  # X-Bewegung zum Wrfel
            0.0,                        # Y-neutral
            0.1 if step < 30 else 0.0,  # Z etwas hoch halten
            0.0                         # Greifer neutral
        ])
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        if step % 10 == 0:
            gripper_pos = obs['observation'][:3]  # Greifer-Position aus Observation
            distance_to_cube = obs['achieved_goal'][4]  # Greifer-Wrfel Distanz
            print(f"  Step {step:2d}: Greifer=[{gripper_pos[0]:.3f}, {gripper_pos[1]:.3f}, {gripper_pos[2]:.3f}], "
                  f"Distanz={distance_to_cube:.3f}, Reward={reward:.2f}")
    
    print("\n PHASE 2: HOVERING SIMULATION (ber dem Wrfel stehen bleiben)")
    print("  Simuliere das Problem: Roboter bleibt ber dem Wrfel ohne Absenkung")
    
    # Hovering simulieren - minimal oder keine Bewegung
    hovering_step = 0
    for step in range(60):  # 60 Steps hovering
        # Sehr kleine zufllige Bewegungen (simuliert stagnierendes Verhalten)
        action = np.array([
            np.random.normal(0, 0.05),  # Kleine X-Bewegung
            np.random.normal(0, 0.05),  # Kleine Y-Bewegung  
            np.random.normal(0, 0.02),  # Minimal Z-Bewegung (kein Absenken!)
            0.0                          # Greifer neutral
        ])
        
        obs, reward, terminated, truncated, info = env.step(action)
        hovering_step += 1
        
        if step % 10 == 0:
            gripper_pos = obs['observation'][:3]
            distance_to_cube = obs['achieved_goal'][4]
            height_diff = gripper_pos[2] - cube_pos[2]
            print(f"  Hovering {step:2d}: Hhe_diff={height_diff:.3f}, "
                  f"Distanz={distance_to_cube:.3f}, Reward={reward:.2f}")
            
            # berprfe ob Anti-Hovering System aktiviert wurde
            if hasattr(env, 'stagnation_detection'):
                hovering_steps = env.stagnation_detection.get('hovering_steps', 0)
                if hovering_steps > 0:
                    print(f"     HOVERING DETECTED: {hovering_steps} steps")
                    
                if hovering_steps > env.stagnation_detection.get('hovering_threshold', 25):
                    print(f"     ANTI-HOVERING INTERVENTION SHOULD BE ACTIVE!")
    
    print("\n PHASE 3: NORMAL TRAINING FORTSETZUNG")
    print("  Lasse das normale Training/Exploration weiterlaufen")
    
    # Normale Aktionen nach Hovering
    for step in range(30):
        action = env.action_space.sample() * 0.5  # Zufllige moderate Aktionen
        obs, reward, terminated, truncated, info = env.step(action)
        
        if step % 10 == 0:
            distance_to_cube = obs['achieved_goal'][4]
            print(f"  Normal {step:2d}: Distanz={distance_to_cube:.3f}, Reward={reward:.2f}")
    
    env.close()
    print("\n ANTI-HOVERING DEBUG TEST ABGESCHLOSSEN!")
    print("\nERWARTETE BEOBACHTUNGEN:")
    print("   Hovering-Erkennung sollte nach ~25 Steps aktiviert werden")
    print("   Anti-Hovering Strafen sollten negatives Reward geben")
    print("   Erzwungene Absenkung sollte Z-negative Aktionen verursachen")
    print("   Verstrkte Belohnungen fr erfolgreiche Absenkung")

if __name__ == '__main__':
    main()