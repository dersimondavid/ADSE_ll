import os
import sys
import math
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import pybullet as p
import pybullet_data
import gymnasium as gym
from gymnasium import spaces
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
        )        # GOAL-BASIERTE Observation (für HER erforderlich) - ERWEITERT
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
        self.max_episode_steps = 1000  # REDUZIERT für schnelleres HER-Lernen        self.cube_grasped = False
        self.success = False
        self.grasp_constraint = None
        self.grasp_constraint_left = None   # Constraint für linken Greiferfinger
        self.grasp_constraint_right = None  # Constraint für rechten Greiferfinger#  ADAPTIVE EXPLORATION & ANTI-LOKALE-OPTIMA SYSTEM - VERBESSERT
        self.exploration_boost = 0.0  # Zustzliche Exploration bei Stagnation
        self.stagnation_detection = {
            'no_improvement_steps': 0,
            'best_distance_to_cube': float('inf'),
            'best_cube_to_target': float('inf'),
            'steps_without_grasp_attempt': 0,
            'exploration_boost_active': False,
            'forced_action_counter': 0,
            'hovering_steps': 0,  # Zählt Steps ber dem Würfel ohne Absenkung
            'hovering_threshold': 3,  # STARK REDUZIERT: Maximale Toleranz für Hovering
            'descent_motivation_active': False,
            'hovering_penalty_escalation': 1.0,  # Multiplier für eskalierende Strafen
            'position_achieved_time': 0,  # Zählt wie lange Position gehalten wird
            'last_height': None,  # Letzte Z-Position für Bewegungserkennung
            'descent_attempts': 0  # Zählt Absenkungsversuche
        }
          # PHASEN-SPEZIFISCHE PARAMETER
        self.current_phase = "approach"  # approach, position, descend, grasp, transport
        self.phase_timeouts = {
            'approach': 200,    # Max Steps für Annäherung
            'position': 150,    # ERWEITERT: Mehr Zeit für präzise Positionierung  
            'descend': 60,      # Etwas mehr Zeit für kontrollierte Absenkung
            'grasp': 40,        # Etwas mehr Zeit für sicheres Greifen
            'transport': 200    # Max Steps für Transport
        }
        self.phase_start_step = 0
        self.phase_success_criteria = {
            'approach': 0.12,   # Abstand zum Würfel (erweitert)
            'position': 0.04,   # XY-Abstand über Würfel (präziser)
            'descend': 0.025,   # Höhe über Würfel (präziser)
            'grasp': True,      # Würfel gegriffen
            'transport': 0.1    # Abstand zum Ziel
        }
        
        # PHASEN-HIERARCHIE FÜR INTELLIGENTE BELOHNUNG
        self.phase_hierarchy = {
            "approach": 1,
            "position": 2, 
            "descend": 3,
            "grasp": 4,
            "transport": 5
        }
        self.phase_history = []  # Verfolge Phasen-Übergänge für Debugging
        
        #  HAUPTZIEL-ORIENTIERTES LERNEN
        self.main_goal_weight = 1.0  # Gewichtung des Hauptziels
        self.sub_goal_decay = 0.95   # Abbau der Unterziel-Belohnungen ber Zeit
        self.episodes_since_success = 0
        self.urgency_factor = 1.0    # Steigt bei wiederholten Fehlschlgen
          # Performance Tracking für Debugging
        self.last_distance = float('inf')
          # NEUE PHASEN-VARIABLEN INITIALISIERUNG - SICHERE DEFAULTS
        self.current_phase = "approach"
        self.phase_start_step = 0
        self.hovering_steps = 0
        self.last_z_pos = 0.5  # Sichere Standard-Höhe statt None
        self.last_xy_distance = float('inf')
        self.hovering_penalty_escalation = 1.0
        
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
        
        # CURRICULUM LEARNING SYSTEM für progressive Schwierigkeit
        self.training_phase = 0  # 0=Easy, 1=Medium, 2=Hard, 3=Expert
        self.success_streak = 0
        self.phase_success_threshold = [0.3, 0.5, 0.7, 0.9]  # Erfolgsraten für Phasen-Upgrade
        self.recent_success_rate = 0.0
        
        # Finger dynamics configuration
        self.left_finger_index = 9   # panda_leftfinger
        self.right_finger_index = 10 # panda_rightfinger
        
        # Configure initial finger dynamics for better grasping
        self._configure_finger_dynamics(self.left_finger_index, 8.0, 6.0)  # Maximale Reibungswerte
        self._configure_finger_dynamics(self.right_finger_index, 8.0, 6.0)  # Maximale Reibungswerte
          # Leichtes Gewicht für bessere Kontrolle
    
    def _configure_finger_dynamics(self, finger_link_index, lateral_friction, spinning_friction):
        # Configure finger dynamics parameters for better grasping
        p.changeDynamics(self.robot_id, finger_link_index,
                        lateralFriction=lateral_friction,  # Dynamic friction based on phase
                        spinningFriction=spinning_friction,  # Dynamic spinning friction
                        frictionAnchor=True,
                        mass=0.1,  # Light weight for better control
                        contactStiffness=100000.0,  # Reduced stiffness for softer contact
                        contactDamping=50.0,  # Reduced damping for smoother contact
                        restitution=0.0,  # No bounce effects
                        linearDamping=1.0,  # Reduced linear damping
                        angularDamping=1.0)  # Reduced angular damping
    
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
        
        # Würfel (0.04 Gre)
        self.cube_id = p.loadURDF("cube.urdf", 
                                  basePosition=[0, 0, 0], 
                                  baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
                                  globalScaling=0.074)
        
        # Kiste (Tray) - FESTE Position
        tray_pos = [0.5, 0.0, 0.02]
        self.tray_id = p.loadURDF("tray/traybox.urdf", 
                                  basePosition=tray_pos, 
                                  baseOrientation=p.getQuaternionFromEuler([0, 0, 0]))
        
        # Roboter-Joint-Informationen
        self.num_joints = p.getNumJoints(self.robot_id)
        self.arm_joint_indices = list(range(7))  # Joints 0-6 für den Arm
        self.gripper_joint_indices = [9, 10]    # Joints 9, 10 für Greifer-Finger
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
        
        # ALLE Joints zurücksetzen
        for i in range(self.num_joints):
            if i < 7:  # Arm-Joints (0-6)
                p.resetJointState(self.robot_id, i, neutral_angles[i])
            elif i in [9, 10]:  # Greifer-Finger (9, 10)
                p.resetJointState(self.robot_id, i, 0.04)  # Voll geöffnet
            else:
                p.resetJointState(self.robot_id, i, 0.0)
        
        # Motor-Control für Arm
        for i in range(7):
            p.setJointMotorControl2(
                self.robot_id, i, p.POSITION_CONTROL,
                targetPosition=neutral_angles[i], 
                force=200,
                maxVelocity=1.0
            )
        
        # Greifer-Motoren - SICHERSTELLEN dass vollständig geöffnet
        for joint_idx in [9, 10]:
            p.setJointMotorControl2(
                self.robot_id, joint_idx, p.POSITION_CONTROL,
                targetPosition=0.04,  # Voll geöffnet
                force=50,
                maxVelocity=0.5
            )
          # Zusätzliche Physik-Steps für sauberen Reset
        for _ in range(10):
            p.stepSimulation()
    
    def reset(self, seed=None, options=None):
        # Seed für Reproduzierbarkeit
        if seed is not None:
            np.random.seed(seed)

        # AGGRESSIVE CONSTRAINT ENTFERNUNG - MEHRFACH VERSUCHEN
        for _ in range(5):  # Erhöht von 3 auf 5 Versuche
            try:
                # Entferne alle bekannten Constraints
                if hasattr(self, 'grasp_constraint') and self.grasp_constraint is not None:
                    p.removeConstraint(self.grasp_constraint)
                    self.grasp_constraint = None
                if hasattr(self, 'grasp_constraint_left') and self.grasp_constraint_left is not None:
                    p.removeConstraint(self.grasp_constraint_left)
                    self.grasp_constraint_left = None
                if hasattr(self, 'grasp_constraint_right') and self.grasp_constraint_right is not None:
                    p.removeConstraint(self.grasp_constraint_right)
                    self.grasp_constraint_right = None
            except Exception as e:
                if self.debug_mode:
                    print(f"Fehler beim Entfernen bekannter Constraints: {e}")

        # Entferne ALLE Constraints, die den Würfel betreffen
        try:
            if hasattr(self, 'cube_id'):
                constraint_ids = []
                for i in range(p.getNumConstraints()):
                    try:
                        constraint_info = p.getConstraintInfo(i)
                        if (constraint_info[2] == self.cube_id or 
                            constraint_info[4] == self.cube_id):
                            constraint_ids.append(i)
                    except:
                        continue

                # Entferne gefundene Constraints
                for constraint_id in constraint_ids:
                    try:
                        p.removeConstraint(constraint_id)
                    except:
                        pass
        except Exception as e:
            if self.debug_mode:
                print(f"Fehler beim Entfernen aller Würfel-Constraints: {e}")

        # Würfel physikalisch "loslassen" - Position und Geschwindigkeit zurücksetzen
        if hasattr(self, 'cube_id'):
            p.resetBaseVelocity(self.cube_id, [0, 0, 0], [0, 0, 0])
            
            # ADAPTIVE CURRICULUM: Würfel-Position basierend auf Lernerfolg
            cube_x, cube_y = self._get_adaptive_cube_position()
            cube_z = 0.05
            
            if self.debug_mode:
                print(f"EPISODE RESET - Phase {self.training_phase} - Würfel bei [{cube_x:.3f}, {cube_y:.3f}]")
            
            # Würfel positionieren
            p.resetBasePositionAndOrientation(
                self.cube_id, 
                [cube_x, cube_y, cube_z], 
                p.getQuaternionFromEuler([0, 0, np.random.uniform(0, 2*np.pi)])
            )

        # Roboter zurücksetzen
        self._reset_robot_position()

        # Greifer explizit öffnen
        for joint_idx in [9, 10]:  # Greifer-Finger
            p.setJointMotorControl2(
                self.robot_id, joint_idx, p.POSITION_CONTROL,
                targetPosition=0.04,  # Vollständig geöffnet
                force=100,
                maxVelocity=0.5
            )

        # Physik stabilisieren - Längere Stabilisierungsphase
        for _ in range(200):  # Erhöht von 100 auf 200
            p.stepSimulation()
            if self.gui:
                time.sleep(1./240.)

        # Episode-Variablen zurücksetzen
        self.episode_step = 0
        self.cube_grasped = False
        self.success = False

        # Phasen-Variablen Reset
        self.current_phase = "approach"
        self.phase_start_step = 0
        self.hovering_steps = 0
        self.last_z_pos = 0.5
        self.last_xy_distance = float('inf')
        self.hovering_penalty_escalation = 1.0

        # Performance Tracking zurücksetzen
        self.last_distance = float('inf')
        self.last_gripper_distance = float('inf')
        self.last_transport_distance = float('inf')
        self.steps_near_cube = 0
        self.last_gripper_to_cube_dist = float('inf')
        self.last_cube_to_target_dist = float('inf')

        # Phasen-Bonus zurücksetzen
        self.phase_bonuses = {
            'exploration': 0.0,
            'approach': 0.0,
            'grasp': 0.0,
            'transport': 0.0,
            'release': 0.0
        }
        self.consecutive_improvements = 0
        self.stagnation_counter = 0

        # Phasen-Historie zurücksetzen
        self.phase_history = []

        # Curriculum Learning Initialisierung (nur beim ersten Reset)
        if not hasattr(self, 'training_phase'):
            self.training_phase = 0
            self.success_streak = 0
            self.phase_success_threshold = [0.3, 0.5, 0.7, 0.9]
            self.recent_success_rate = 0.0
            self.recent_successes = []

        # Event-basierte Belohnungs-Tracker zurücksetzen
        self.was_in_auto_grasp_zone = False
        self.contact_achieved = False
        self.last_xy_distance = float('inf')
        self.last_z_pos = None
        self.last_close_distance = float('inf')

        # Finale Physik-Simulation für Stabilität
        for _ in range(50):
            p.stepSimulation()
            if self.gui:
                time.sleep(1./240.)

        obs = self._get_obs()
        info = {'is_success': False}

        return obs, info
    
    def _get_obs(self):
        try:
            # 1. End-Effector Position (3 Werte)
            end_effector_state = p.getLinkState(self.robot_id, self.end_effector_link_index)
            end_effector_pos = np.array(end_effector_state[0])
            
            # NaN-Check für End-Effector
            if np.any(np.isnan(end_effector_pos)) or np.any(np.isinf(end_effector_pos)):
                print(f" NaN/Inf in end_effector_pos: {end_effector_pos}")
                end_effector_pos = np.array([0.3, 0.0, 0.4])  # Safe fallback
            
            # 2. Greifer-Zustand (2 Werte)
            try:
                gripper_pos_left = p.getJointState(self.robot_id, 9)[0]
                gripper_pos_right = p.getJointState(self.robot_id, 10)[0]
                
                # NaN-Check für Greifer
                if np.isnan(gripper_pos_left) or np.isnan(gripper_pos_right):
                    print(f" NaN in gripper positions: left={gripper_pos_left}, right={gripper_pos_right}")
                    gripper_pos_left = gripper_pos_right = 0.03  # Safe fallback
                    
                avg_gripper_pos = (gripper_pos_left + gripper_pos_right) / 2.0
                
            except Exception as e:
                print(f" Greifer-Zustand Fehler: {e}")
                gripper_pos_left = gripper_pos_right = avg_gripper_pos = 0.03
            
            # 3. Würfel Position und Orientierung (4 Werte)
            try:
                cube_pos, cube_orn = p.getBasePositionAndOrientation(self.cube_id)
                cube_pos = np.array(cube_pos)
                
                # NaN-Check für Würfel
                if np.any(np.isnan(cube_pos)) or np.any(np.isinf(cube_pos)):
                    print(f" NaN/Inf in cube_pos: {cube_pos}")
                    cube_pos = np.array([0.0, 0.0, 0.05])  # Safe fallback
                
                # Euler-Winkel für kompakte Orientierung (nur Z-Rotation ist relevant)
                cube_euler = p.getEulerFromQuaternion(cube_orn)
                cube_z_rotation = cube_euler[2]  # Rotation um Z-Achse
                
                if np.isnan(cube_z_rotation) or np.isinf(cube_z_rotation):
                    cube_z_rotation = 0.0
                    
            except Exception as e:
                print(f" Würfel-Position Fehler: {e}")
                cube_pos = np.array([0.0, 0.0, 0.05])
                cube_z_rotation = 0.0
            
            # 4. Relative Vektoren (6 Werte) - mit NaN-Schutz
            end_effector_to_cube = cube_pos - end_effector_pos
            end_effector_to_target = self.target_position - end_effector_pos
            cube_to_target = self.target_position - cube_pos
            
            # NaN-Check für relative Vektoren
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
              # 5. KONTAKT-INFORMATIONEN (3 Werte) - WICHTIG für GREIFEN!
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
            
            # Final NaN-Check für Distanz
            if np.isnan(gripper_to_cube_dist) or np.isinf(gripper_to_cube_dist):
                gripper_to_cube_dist = 1.0  # Safe fallback
            
            # Observation zusammensetzen (3+3+4+3+3+3+3+2 = 24 Werte)
            observation = np.concatenate([
                end_effector_pos,           # 3 - End-Effector Position 
                [gripper_pos_left, gripper_pos_right, avg_gripper_pos],  # 3 - Detaillierte Greifer-Info
                cube_pos, [cube_z_rotation], # 4 - Würfel Position + Orientierung
                end_effector_to_cube,       # 3 - Vektor: Greifer -> Würfel
                end_effector_to_target,     # 3 - Vektor: Greifer -> Ziel (WICHTIG!)
                cube_to_target,             # 3 - Vektor: Würfel -> Ziel
                [contact_left, contact_right, bilateral_contact], # 3 - KONTAKT-INFO!
                [task_progress, gripper_to_cube_dist]  # 2 - Task Status + Distanz
            ]).astype(np.float32)
            
            # KRITISCHER NaN-Check für finale Observation
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
                cube_pos[0], cube_pos[1], cube_pos[2],  # Würfel Position
                is_grasped_binary,                       # Greif-Status (0/1)
                gripper_to_cube_dist                     # Distanz Greifer->Würfel
            ]).astype(np.float32)
            
            # NaN-Check für achieved_goal
            if np.any(np.isnan(achieved_goal)) or np.any(np.isinf(achieved_goal)):
                print(f" NaN/Inf in achieved_goal: {achieved_goal}")
                achieved_goal = np.array([0.0, 0.0, 0.05, 0.0, 1.0], dtype=np.float32)
            
            # desired_goal = [target_x, target_y, target_z, should_be_grasped, optimal_gripper_dist]
            desired_goal = np.array([
                self.target_position[0], self.target_position[1], self.target_position[2],  # Ziel-Position
                1.0,  # Sollte gegriffen werden
                0.0   # Optimale Greifer-Distanz (0 = perfekt gegriffen)
            ]).astype(np.float32)
            
            # NaN-Check für desired_goal
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
                'observation': np.zeros(24, dtype=np.float32),  # Angepasst für neue Dimension
                'achieved_goal': np.array([0.0, 0.0, 0.05, 0.0, 1.0], dtype=np.float32),
                'desired_goal': np.array([0.5, 0.0, 0.08, 1.0, 0.0], dtype=np.float32),
            }

    def _update_grasp_state(self):
        # Hole aktuelle Positionen und Orientierungen
        end_effector_pos = np.array(p.getLinkState(self.robot_id, self.end_effector_link_index)[0])
        end_effector_orn = p.getLinkState(self.robot_id, self.end_effector_link_index)[1]
        cube_pos, cube_orn = p.getBasePositionAndOrientation(self.cube_id)
        cube_pos = np.array(cube_pos)
        
        # Berechne Distanzen
        distance_to_cube = np.linalg.norm(end_effector_pos - cube_pos)
        xy_distance = np.linalg.norm(end_effector_pos[:2] - cube_pos[:2])
        height_above_cube = end_effector_pos[2] - cube_pos[2]
        
        # Hole aktuelle Greifer-Position
        left_finger_pos = p.getJointState(self.robot_id, self.left_finger_index)[0]
        right_finger_pos = p.getJointState(self.robot_id, self.right_finger_index)[0]
        avg_gripper_pos = (left_finger_pos + right_finger_pos) / 2
        
        # Prüfe Kontakte
        left_contact = len(p.getContactPoints(self.robot_id, self.cube_id, self.left_finger_index)) > 0
        right_contact = len(p.getContactPoints(self.robot_id, self.cube_id, self.right_finger_index)) > 0
        
        # Berechne die Transformation vom Welt- zum End-Effector-Koordinatensystem
        ee_matrix = np.reshape(p.getMatrixFromQuaternion(end_effector_orn), (3, 3))
        cube_matrix = np.reshape(p.getMatrixFromQuaternion(cube_orn), (3, 3))
        
        # Berechne die relative Ausrichtung zwischen End-Effector und Würfel
        relative_orn = cube_matrix @ ee_matrix.T
        
        # Würfel-Parameter
        cube_width = 0.05  # Breite des Würfels
        finger_offset = 0.002  # Kleiner Offset für besseren Kontakt
        
        # Greifer-Logik nach Phase
        if distance_to_cube < 0.08 and self.grasp_constraint is None:
            # Annäherung oder Absenken: Greifer weit öffnen
            if self.current_phase in ["approach", "position", "descend"]:
                target_open = 0.15
                for joint_idx in [self.left_finger_index, self.right_finger_index]:
                    p.setJointMotorControl2(
                        self.robot_id, joint_idx, p.POSITION_CONTROL,
                        targetPosition=target_open,
                        force=50, maxVelocity=0.05, positionGain=0.5, velocityGain=0.1)
            # Erst in der grasp-Phase präzise schließen
            elif self.current_phase == "grasp":
                # Low friction for initial contact
                self._configure_finger_dynamics(self.left_finger_index, 0.001, 0.001)
                self._configure_finger_dynamics(self.right_finger_index, 0.001, 0.001)
                # Calculate exact cube edge positions
                left_edge = -(cube_width/2) * relative_orn[0,0]
                right_edge = (cube_width/2) * relative_orn[0,0]
                # Set finger positions with precise control
                p.setJointMotorControlArray(
                    self.robot_id,
                    [self.left_finger_index, self.right_finger_index],
                    p.POSITION_CONTROL,
                    targetPositions=[left_edge, right_edge],
                    forces=[50, 50],
                    positionGains=[0.5, 0.5],
                    velocityGains=[0.1, 0.1]
                )
                # Wait for finger movement
                for _ in range(15):
                    p.stepSimulation()
                    if self.gui:
                        time.sleep(1./240.)
                # After contact, increase friction for stable grasp
                self._configure_finger_dynamics(self.left_finger_index, 100.0, 10.0)
                self._configure_finger_dynamics(self.right_finger_index, 100.0, 10.0)
                
            # Kontakt-Belohnung
            try:
                contacts_left = p.getContactPoints(bodyA=self.robot_id, bodyB=self.cube_id, linkIndexA=9)
                contacts_right = p.getContactPoints(bodyA=self.robot_id, bodyB=self.cube_id, linkIndexA=10)
                has_contact = len(contacts_left) > 0 or len(contacts_right) > 0
                
                if has_contact and not hasattr(self, 'contact_achieved'):
                    total_reward += 25.0
                    phase_info += "FIRST_CONTACT(+25) "
                    self.contact_achieved = True
                elif not has_contact:
                    self.contact_achieved = False
            except Exception:
                pass
                
        # Halte die Greifposition konstant, solange das Constraint existiert
        elif self.grasp_constraint is not None:
            # Hole aktuelle Endeffektor-Position und -Ausrichtung
            ee_state = p.getLinkState(self.robot_id, self.end_effector_link_index)
            ee_pos = np.array(ee_state[0])
            ee_orn = np.array(ee_state[1])
            ee_matrix = np.array(p.getMatrixFromQuaternion(ee_orn)).reshape(3, 3)
            
            # Feste Zielpositionen für die Greiffinger relativ zum Endeffektor
            # Diese Werte sind so gewählt, dass sie den Würfel symmetrisch greifen
            left_finger_target = -(cube_width/2) * relative_orn[0,0]  # 20mm nach links
            right_finger_target = (cube_width/2) * relative_orn[0,0]  # 20mm nach rechts
            
            # Setze die Greiffinger auf die Zielpositionen mit hoher Kraft
            p.setJointMotorControlArray(
                self.robot_id,
                [self.left_finger_index, self.right_finger_index],
                p.POSITION_CONTROL,
                targetPositions=[left_finger_target, right_finger_target],
                forces=[1000, 1000],  # Erhöhte Kraft für stabileren Griff
                positionGains=[2.0, 2.0],  # Erhöhte Steifigkeit
                velocityGains=[0.1, 0.1]  # Geringe Dämpfung für schnelleres Schließen
            )
        
        # LOSLASSEN: Greifer weit geöffnet ODER strategisches Loslassen in der Kiste
        elif avg_gripper_pos > 0.025:
            # Prüfe ob wir über der Kiste sind für kontrolliertes Loslassen
            cube_to_target_dist = np.linalg.norm(cube_pos - self.target_position)
            
            if cube_to_target_dist < 0.12:  # Nah genug an der Kiste
                # Sanftes Öffnen der Greifer vor dem Loslassen
                for joint_idx in [self.left_finger_index, self.right_finger_index]:
                    p.setJointMotorControl2(
                        self.robot_id, joint_idx, p.POSITION_CONTROL,
                        targetPosition=0.04,  # Vollständig geöffnet
                        force=100,  # Sanfte Kraft
                        maxVelocity=0.05  # Langsame Bewegung
                    )
                
                # Kurz warten für sanftes Öffnen
                for _ in range(15):
                    p.stepSimulation()
                    if self.gui:
                        time.sleep(1./240.)
                # Entferne alle Constraints
                if self.grasp_constraint is not None:
                    p.removeConstraint(self.grasp_constraint)
                    self.grasp_constraint = None
                self.cube_grasped = False
                
                # Setze Reibung zurück für sanftes Zuklemmen
                self._configure_finger_dynamics(self.left_finger_index, 0.001, 0.001)
                self._configure_finger_dynamics(self.right_finger_index, 0.001, 0.001)
                
                if self.debug_mode:
                    print("🎯 STRATEGISCHES LOSLASSEN in der Kiste!")

    # GoalEnv Interface für HER-Kompatibilitt
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
    
    # GoalEnv Interface für HER-Kompatibilitt
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
        
        # KRITISCHER NaN-Check für Input-Action
        if not isinstance(action, np.ndarray):
            action = np.array(action, dtype=np.float32)
        
        if np.any(np.isnan(action)) or np.any(np.isinf(action)):
            print(f" KRITISCH: NaN/Inf in Action: {action}")
            action = np.zeros(4, dtype=np.float32)  # Safe fallback
        
          # Aktionen interpretieren: [delta_x, delta_y, delta_z, gripper_command]
        position_delta = action[:3] * 0.03  # REDUZIERT: 3cm max. Bewegung für mehr Kontrolle
        gripper_command = action[3]
        
        # Zustzliche Sicherheit für Aktionen
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
            
            # Kollisionsvermeidung: Nicht zu nah an Kiste auer wenn Würfel gegriffen
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
        
        # ERWEITERTE Arbeitsraum-Limits für bessere Sicherheit
        target_pos[0] = np.clip(target_pos[0], -0.6, 0.8)   # X-Limits 
        target_pos[1] = np.clip(target_pos[1], -0.6, 0.6)   # Y-Limits  
        target_pos[2] = np.clip(target_pos[2], 0.035, 0.7)   # Z-Limits 
        
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
        
        # Greifer-Kontrolle
        current_gripper_pos = p.getJointState(self.robot_id, 9)[0]
        gripper_target = current_gripper_pos
        
        # Hole aktuelle Positionen für Distanzberechnung
        end_effector_pos = np.array(p.getLinkState(self.robot_id, self.end_effector_link_index)[0])
        cube_pos, _ = p.getBasePositionAndOrientation(self.cube_id)
        cube_pos = np.array(cube_pos)
        
        # Berechne Distanzen
        distance_to_cube = np.linalg.norm(end_effector_pos - cube_pos)
        xy_distance = np.linalg.norm(end_effector_pos[:2] - cube_pos[:2])
        height_above_cube = end_effector_pos[2] - cube_pos[2]
        
        # Standardisierte Greifer-Positionierung
        if distance_to_cube < 0.15:  # Wenn nah am Würfel
            if xy_distance < 0.025:
                if height_above_cube > 0.012:  # Noch über dem Würfel: immer weit öffnen
                    gripper_target = 0.2
                elif height_above_cube <= 0.03:  # Direkt über dem Würfel: schließen
                    gripper_target = 0.025  # Feste, optimale Position für Würfelgriff
            else:
                # Normale Steuerung wenn nicht in Greifposition
                if gripper_command > 0.3:  # Schließen
                    gripper_target = max(0.0, gripper_target - 0.01)
                elif gripper_command < -0.3:  # Öffnen
                    gripper_target = min(0.04, gripper_target + 0.01)
        else:
            # Normale Steuerung wenn weit vom Würfel entfernt
            if gripper_command > 0.3:  # Schließen
                gripper_target = max(0.0, gripper_target - 0.01)
            elif gripper_command < -0.3:  # Öffnen
                gripper_target = min(0.04, gripper_target + 0.01)
        
        # Beide Greifer-Finger synchron steuern mit konsistenter Kraft
        for joint_idx in [self.left_finger_index, self.right_finger_index]:
            p.setJointMotorControl2(
                self.robot_id, joint_idx, p.POSITION_CONTROL,
                targetPosition=gripper_target, 
                force=50,  # Reduced force for smoother control
                maxVelocity=0.02,  # Slow movement for precision
                positionGain=0.5,  # Reduced position gain
                velocityGain=0.1  # Reduced velocity gain
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
          # Aktuelle Würfel position für Debug
        cube_pos, _ = p.getBasePositionAndOrientation(self.cube_id)
        cube_pos = np.array(cube_pos)
        
        info = {
            'is_success': success,
            'cube_grasped': self.cube_grasped,
            'episode_step': self.episode_step,
            'cube_to_target_distance': np.linalg.norm(cube_pos - self.target_position),  # DEBUG
            'training_phase': self.training_phase,  # Curriculum Info
            'success_rate': getattr(self, 'recent_success_rate', 0.0),  # Aktuelle Erfolgsrate
        }          #  CURRICULUM UPDATE bei Episode-Ende
        if terminated or truncated:
            self._update_curriculum_progress(success)
            
            # DEBUG: Phasen-Historie bei Episode-Ende ausgeben
            if self.debug_mode and hasattr(self, 'phase_history') and len(self.phase_history) > 0:
                print(f"\n📊 PHASEN-HISTORIE Episode beendet (Erfolg: {'✅' if success else '❌'}):")
                total_progress_reward = 0
                total_regression_penalty = 0
                for step, from_phase, to_phase, level_change in self.phase_history:
                    direction = "📈" if level_change > 0 else "📉" if level_change < 0 else "➡️"
                    if level_change > 0:
                        total_progress_reward += abs(level_change)
                    elif level_change < 0:
                        total_regression_penalty += abs(level_change)
                    print(f"  Step {step:3d}: {from_phase} -> {to_phase} {direction} (Δ{level_change:+d})")
                
                progress_ratio = total_progress_reward / max(1, total_progress_reward + total_regression_penalty)
                print(f"  Fortschritt-Quote: {progress_ratio:.1%} (Fortschritt: {total_progress_reward}, Rückfall: {total_regression_penalty})")
        
        return obs, reward, terminated, truncated, info
    

    def _evaluate_phase_transition(self, previous_phase, new_phase):
        
        # Sichere Phasen-Abruf mit Fallback
        prev_level = self.phase_hierarchy.get(previous_phase, 1)
        new_level = self.phase_hierarchy.get(new_phase, 1)
        level_change = new_level - prev_level

        # Phasen-Historie aktualisieren
        if not hasattr(self, 'phase_history'):
            self.phase_history = []
        self.phase_history.append((self.episode_step, previous_phase, new_phase, level_change))
        # Behalte nur letzte 10 Transitions für Debugging
        if len(self.phase_history) > 10:
            self.phase_history.pop(0)

        # Wiederholte Wechsel zwischen denselben Phasen bestrafen
        repeat_penalty = 0.0
        if len(self.phase_history) >= 2:
            last = self.phase_history[-2]
            if last[1] == new_phase and last[2] == previous_phase:
                # Hin- und Herwechseln zwischen zwei Phasen
                repeat_penalty = -2.0  # Kleine Strafe für Phase-Bouncing

        if level_change > 0:
            # FORTSCHRITT: Positive Belohnung
            progress_reward = 0.0 + (level_change * 0)  # Reduzierte Grundbelohnung

            # Reduzierte Boni für kritische Übergänge
            bonus = 0.0
            if previous_phase == "position" and new_phase == "descend":
                bonus = 0  # Reduzierter Bonus
            elif previous_phase == "descend" and new_phase == "grasp":
                bonus = 0  # Deutlich reduzierter Bonus
            elif previous_phase == "grasp" and new_phase == "transport":
                bonus = 0  # Deutlich reduzierter Bonus

            total_reward = progress_reward + bonus + repeat_penalty
            info = f"PHASE_PROGRESS({previous_phase}->{new_phase},+{total_reward:.1f}) "

            if self.debug_mode:
                pass
                #print(f"  ✅ PHASEN-FORTSCHRITT: {previous_phase}(Lv{prev_level}) -> {new_phase}(Lv{new_level}) = +{total_reward:.1f}")

        elif level_change < 0:
            # RÜCKFALL: Negative Belohnung (Bestrafung)
            regression_penalty = 0 + (level_change * 0)  # Größere Rückfälle = mehr Strafe

            # Besonders schwere Strafen für kritische Rückfälle
            penalty = 0.0
            if new_phase == "approach" and prev_level > 2:
                penalty = 0  # Zurück zur Annäherung ist schlecht
            elif new_phase == "position" and previous_phase == "grasp":
                penalty = 0  # Würfel verloren nach Griff!
            elif new_phase == "approach" and previous_phase == "transport":
                penalty = 0  # Katastrophaler Rückfall: Würfel verloren beim Transport

            total_reward = regression_penalty + penalty + repeat_penalty
            info = f"PHASE_REGRESSION({previous_phase}->{new_phase},{total_reward:.1f}) "

            if self.debug_mode:
                pass
                #print(f"  ❌ PHASEN-RÜCKFALL: {previous_phase}(Lv{prev_level}) -> {new_phase}(Lv{new_level}) = {total_reward:.1f}")

        else:
            # Keine Phasen-Änderung (sollte nicht aufgerufen werden)
            total_reward = 0.0 + repeat_penalty
            info = f"PHASE_SAME({new_phase},{total_reward:.1f}) "

        return total_reward, info

    def compute_reward(self, achieved_goal, desired_goal, info):
        # Sicherstellen, dass achieved_goal und desired_goal numpy arrays sind
        achieved_goal = np.array(achieved_goal, dtype=np.float32)
        desired_goal = np.array(desired_goal, dtype=np.float32)        
        # Batch-Processing für HER
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
        
        # NaN-Checks für berechnete Werte
        if np.isnan(cube_to_target_dist) or np.isinf(cube_to_target_dist):
            cube_to_target_dist = 1.0
        if np.isnan(gripper_to_cube_dist) or np.isinf(gripper_to_cube_dist):
            gripper_to_cube_dist = 1.0
        
        total_reward = 0.0
        phase_info = ""
          # Aktueller Position bestimmen mit sicherer Initialisierung
        try:
            end_effector_pos = np.array(p.getLinkState(self.robot_id, self.end_effector_link_index)[0])
            if np.any(np.isnan(end_effector_pos)):
                end_effector_pos = np.array([0, 0, 0.5])
        except Exception:
            end_effector_pos = np.array([0, 0, 0.5])
            
        xy_distance = np.linalg.norm(end_effector_pos[:2] - cube_pos[:2])
        height_above_cube = end_effector_pos[2] - cube_pos[2]
        
        # Sichere Initialisierung der Tracking-Variablen
        if not hasattr(self, 'last_z_pos') or self.last_z_pos is None:
            self.last_z_pos = end_effector_pos[2]
        if not hasattr(self, 'last_xy_distance'):
            self.last_xy_distance = xy_distance
          # PHASEN-MANAGEMENT: Automatische Phasen-Erkennung und -Übergänge
        previous_phase = self.current_phase
        self._update_current_phase(gripper_to_cube_dist, xy_distance, height_above_cube, is_grasped, cube_to_target_dist)
        
        # INTELLIGENTER PHASEN-WECHSEL: Fortschritt belohnen, Rückfall bestrafen
        if previous_phase != self.current_phase:
            phase_reward, phase_change_info = self._evaluate_phase_transition(previous_phase, self.current_phase)
            total_reward += phase_reward
            phase_info += phase_change_info
            self.phase_start_step = self.episode_step
          # PHASE-SPEZIFISCHE BELOHNUNGEN
        if self.current_phase == "approach":
            # Nur Verbesserung der Annäherung belohnen
            if hasattr(self, 'last_gripper_to_cube_dist'):
                if gripper_to_cube_dist < self.last_gripper_to_cube_dist - 0.01:
                    improvement = self.last_gripper_to_cube_dist - gripper_to_cube_dist
                    total_reward += improvement * 20.0
                    phase_info += f"APPROACHING({improvement:.3f}) "
                    
        elif self.current_phase == "position":
            # XY-Positionierung über dem Würfel
            if hasattr(self, 'last_xy_distance'):
                if xy_distance < self.last_xy_distance - 0.005:
                    xy_improvement = self.last_xy_distance - xy_distance
                    total_reward += xy_improvement * 30.0
                    phase_info += f"POSITIONING({xy_improvement:.3f}) "
            self.last_xy_distance = xy_distance
              # KRITISCH: Anti-Hovering Mechanismus
            #hover_reward, hover_info = self._apply_anti_hovering_penalty(xy_distance, height_above_cube, end_effector_pos)
            #total_reward += hover_reward
            #phase_info += hover_info
            
        elif self.current_phase == "descend":
            # Absenkung wird stark belohnt
            if hasattr(self, 'last_z_pos') and self.last_z_pos is not None:
                z_movement = self.last_z_pos - end_effector_pos[2]  # Positiv = Absenkung
                if z_movement > 0.002:  # Signifikante Absenkung
                    descent_reward = z_movement * 300.0  # SEHR STARKE Belohnung für Absenkung
                    total_reward += descent_reward
                    phase_info += f"DESCENDING({descent_reward:.1f}) "
                elif z_movement < -0.0001:  # Aufwärtsbewegung bestraft
                    total_reward -= 20.0
                    phase_info += "GOING_UP(-20) "
            
            # Sichere Z-Position Update
            self.last_z_pos = end_effector_pos[2]
            
            # Timeout-Strafe für zu langsame Absenkung
            phase_duration = self.episode_step - self.phase_start_step
            if phase_duration > self.phase_timeouts['descend']:
                total_reward -= 5.0  # Strafe für Zeitüberschreitung
                phase_info += "DESCEND_TIMEOUT(-5) "
                
        elif self.current_phase == "grasp":
            # WEITER ABSENKEN, solange nicht gegriffen
            if not is_grasped and hasattr(self, 'last_z_pos') and self.last_z_pos is not None:
                z_movement = self.last_z_pos - end_effector_pos[2]  # Positiv = Absenkung
                if z_movement > 0.001:  # Signifikante Absenkung belohnen
                    descent_reward = z_movement * 100.0  # Starke Belohnung
                    total_reward += descent_reward
                    phase_info += f"GRASP_DESCENT({descent_reward:.1f}) "
                elif z_movement < -0.0001:  # Aufwärtsbewegung bestraft (wie in descend)
                    total_reward -= 20.0
                    phase_info += "GOING_UP(-20) "

            # Hovering-Bestrafung wie in position-Phase
            #hover_reward, hover_info = self._apply_anti_hovering_penalty(xy_distance, height_above_cube, end_effector_pos)
            #total_reward += hover_reward
            #phase_info += hover_info

            # Z-Position für nächsten Step aktualisieren
            self.last_z_pos = end_effector_pos[2]

            # Greifen wird stark belohnt
            if is_grasped and not hasattr(self, 'grasp_achieved'):
                total_reward += 50.0  # Massive Belohnung für erfolgreiches Greifen
                phase_info += "SUCCESSFULLY_GRASPED(+50) "
                self.grasp_achieved = True
            elif not is_grasped:
                self.grasp_achieved = False
                
            # Kontakt-Belohnung
            try:
                contacts_left = p.getContactPoints(bodyA=self.robot_id, bodyB=self.cube_id, linkIndexA=9)
                contacts_right = p.getContactPoints(bodyA=self.robot_id, bodyB=self.cube_id, linkIndexA=10)
                has_contact = len(contacts_left) > 0 or len(contacts_right) > 0
                
                if has_contact and not hasattr(self, 'contact_achieved'):
                    total_reward += 25.0
                    phase_info += "FIRST_CONTACT(+25) "
                    self.contact_achieved = True
                elif not has_contact:
                    self.contact_achieved = False
            except Exception:
                pass
                
        elif self.current_phase == "transport":
            # Transport zum Ziel
            if hasattr(self, 'last_cube_to_target_dist'):
                if cube_to_target_dist < self.last_cube_to_target_dist - 0.01:
                    transport_improvement = self.last_cube_to_target_dist - cube_to_target_dist
                    total_reward += transport_improvement * 40.0
                    phase_info += f"TRANSPORTING({transport_improvement:.3f}) "
            
            # Erfolgsbelohnung
            if cube_to_target_dist < self.distance_threshold:
                total_reward += 100.0  # Massive Erfolgsbelohnung
                phase_info += "MISSION_SUCCESS(+100) "        
        # GLOBALE STRAFEN
        collision_reward, collision_info = self._apply_collision_penalties()
        total_reward += collision_reward
        phase_info += collision_info
        
        # Tracking-Werte aktualisieren
        self.last_gripper_to_cube_dist = gripper_to_cube_dist
        self.last_cube_to_target_dist = cube_to_target_dist
        
        # Debug-Ausgabe
        if self.debug_mode and self.episode_step % 50 == 0:
            print(f"Step {self.episode_step:3d}: Phase={self.current_phase} | Reward={total_reward:+6.2f} | {phase_info}")
            print(f"  DistCube: {gripper_to_cube_dist:.3f}m | XY: {xy_distance:.3f}m | Height: {height_above_cube:.3f}m")
        
        # Reward begrenzen und zurückgeben
        total_reward = np.clip(total_reward, -50.0, 150.0)
        return float(total_reward)
    
    def _update_current_phase(self, gripper_to_cube_dist, xy_distance, height_above_cube, is_grasped, cube_to_target_dist):
        if is_grasped and cube_to_target_dist > self.distance_threshold:
            self.current_phase = "transport"
        elif is_grasped:
            self.current_phase = "transport"
        # Grasp: Sehr exakt über dem Würfel (an der Kante) und sehr nah dran
        elif gripper_to_cube_dist < 0.045 and xy_distance < 0.015 and height_above_cube <= 0.04:
            self.current_phase = "grasp"
        # Descend: Gut über dem Würfel positioniert, im Absenkbereich
        elif gripper_to_cube_dist < 0.065 and xy_distance < 0.02 and 0.035 < height_above_cube <= 0.08:
            self.current_phase = "descend"
        # Position: Annäherung an optimale XY-Position über dem Würfel (ERWEITERT)
        elif gripper_to_cube_dist <= 0.12 and xy_distance < 0.04 and height_above_cube > 0.08:
            self.current_phase = "position"
        # Approach: Allgemeine Annäherung an den Würfel
        elif gripper_to_cube_dist < 0.2:
            self.current_phase = "approach"
    

    def _apply_anti_hovering_penalty(self, xy_distance, height_above_cube, end_effector_pos):
        # VERSTÄRKTE Anti-Hovering Mechanismen
        
        # Sichere Initialisierung mit Null-Checks
        if not hasattr(self, 'hovering_steps'):
            self.hovering_steps = 0
        if not hasattr(self, 'last_z_pos') or self.last_z_pos is None:
            self.last_z_pos = end_effector_pos[2]
        if not hasattr(self, 'hovering_penalty_escalation'):
            self.hovering_penalty_escalation = 1.0
            
        # Hovering-Erkennung: Gut positioniert aber zu hoch und bewegt sich nicht nach unten
        is_well_positioned = xy_distance < 0.08
        is_too_high = height_above_cube > 0.10
        
        # Sichere Bewegungserkennung mit Null-Check
        if self.last_z_pos is not None:
            is_descending = (end_effector_pos[2] < self.last_z_pos - 0.003)
        else:
            is_descending = False
            self.last_z_pos = end_effector_pos[2]
        
        total_reward = 0.0
        phase_info = ""
        
        if is_well_positioned and is_too_high:
            self.hovering_steps += 1
            
            # ESKALIERENDE STRAFEN für anhaltendes Hovering
            if self.hovering_steps > self.stagnation_detection['hovering_threshold']:
                # Quadratische Eskalation der Strafen
                escalation = min(self.hovering_steps / 5.0, 10.0)  # Bis zu 10x Verstärkung
                base_penalty = -8.0 * escalation  # Bis zu -80 pro Step!
                total_reward -= abs(base_penalty)
                
                # Extra Strafe für hartnäckiges Hovering
                if self.hovering_steps > 15:
                    total_reward -= 25.0  # Massive Zusatzstrafe
                    
                phase_info += f"SEVERE_HOVERING_PENALTY({base_penalty:.1f}) "
            
            # MASSIVE Belohnung für Absenkung während Hovering
            if is_descending:
                urgency_bonus = 30.0 + (self.hovering_steps * 2.0)  # Eskalierender Bonus
                total_reward += urgency_bonus
                phase_info += f"DESCENT_RESCUE_BONUS({urgency_bonus:.1f}) "
                self.hovering_steps = max(0, self.hovering_steps - 3)  # Reduziere Hovering-Counter
        else:
            # Reset Hovering-Counter wenn Position verlassen
            self.hovering_steps = max(0, self.hovering_steps - 1)
            
        self.last_z_pos = end_effector_pos[2]
        return total_reward, phase_info
        
    def _apply_collision_penalties(self):
       # Kollisionsvermeidung
        total_reward = 0.0
        phase_info = ""
        try:
            if not hasattr(self, 'cube_grasped') or not self.cube_grasped:
                tray_contacts = p.getContactPoints(bodyA=self.robot_id, bodyB=self.tray_id)
                if len(tray_contacts) > 0:
                    total_reward -= 5.0  # Strafe für Kollision
                    phase_info += "COLLISION(-5) "
        except Exception:
            pass
        return total_reward, phase_info
        
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
            
            # Erfolg = Würfel wurde erfolgreich zur Kiste transportiert
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
        
        # Gespiegelte Position: Tray bei +0.5, Würfel-Spawn-Bereich bei -0.5
        spawn_center_x = -tray_center_x  # -0.5 (gespiegelt)
        spawn_center_y = tray_center_y   # 0.0 (gleiche Y-Position)
        
        # Würfel spawnt in exakt der gleichen Flche wie die Tray-Box, nur gespiegelt
        cube_x = np.random.uniform(spawn_center_x - tray_size, spawn_center_x + tray_size)  # [-0.65, -0.35]
        cube_y = np.random.uniform(spawn_center_y - tray_size, spawn_center_y + tray_size)  # [-0.15, +0.15]
        
        if self.debug_mode:
            print(f" Würfel spawnt bei: [{cube_x:.3f}, {cube_y:.3f}] - Gespiegelte Tray-Flche")
        
        return cube_x, cube_y
    
    
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
                len(self.recent_successes) >= 10):  # Mindestens 10 Episodes für valide Rate
                
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
            
            # Reset für nchste Episode
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
        
        plot_path = os.path.join(self.save_path, f'sac_her_training_plot_{self.n_calls}')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Plot gespeichert: {plot_path}")


def train_sac_her_model(gui=True, total_timesteps=1000000, pretrained_model_path=None):
    print(" Starte SAC + HER Training für Pick-and-Place...")
    print(" NEW FEATURES:")
    print("   SAC: Off-Policy Algorithmus mit kontinuierlichen Aktionen")
    print("   HER: Hindsight Experience Replay für sparse rewards")
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
    
    # für SAC muss die Environment in DummyVecEnv gewrappt werden
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
            "MultiInputPolicy",  # für Goal-basierte Environments
            env, 
            replay_buffer_class=HerReplayBuffer,
            replay_buffer_kwargs=dict(
                n_sampled_goal=4,  # Anzahl der HER-Goals pro echtem Goal
                goal_selection_strategy='future',  # Nimm zuknftige Positionen als Goals
            ),
            verbose=1,
            learning_rate=3e-4,      # Standard Learning Rate für stabile Exploration
            buffer_size=1000000,     # Groer Replay Buffer für diverse Erfahrungen
            learning_starts=10000,   # MEHR Exploration vor Training-Start
            batch_size=128,          # Kleinere Batches für stabilere Updates
            tau=0.005,               # Langsame Target Updates
            gamma=0.99,              # HHER für langfristige Belohnungen
            train_freq=4,            # Trainiere alle 4 Steps (weniger frequent)
            gradient_steps=1,        
            ent_coef='auto',         # KRITISCH: Automatische Entropie für Exploration
            target_update_interval=1, 
            tensorboard_log="./sac_her_tensorboard/",
            policy_kwargs=dict(
                net_arch=[512, 512, 256],  # Groes Netzwerk für komplexe Policies
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
        # Model mit Environment-Referenz laden (für HER erforderlich)
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
    print("    Sparse Rewards: Perfekt für Pick-and-Place Tasks")
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
            train_sac_her_model(gui=True, total_timesteps=500000)  # Weniger Steps für ersten Test
            
        elif choice == '2':
            print(" Starte NEUES SAC+HER Training ohne GUI...")
            train_sac_her_model(gui=False, total_timesteps=10000000)  # Mehr Steps für Volltraining
            
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
    
    # Test-Sequenz: Roboter ber Würfel positionieren
    print("\n PHASE 1: Roboter ber Würfel positionieren")
    
    cube_pos = obs['achieved_goal'][:3]
    print(f" Würfel-Position: [{cube_pos[0]:.3f}, {cube_pos[1]:.3f}, {cube_pos[2]:.3f}]")
    
    # Bewegung ber den Würfel (horizontal)
    for step in range(50):
        # Bewegung zum Würfel (horizontal)
        action = np.array([
            0.3 if step < 20 else 0.0,  # X-Bewegung zum Würfel
            0.0,                        # Y-neutral
            0.1 if step < 30 else 0.0,  # Z etwas hoch halten
            0.0                         # Greifer neutral
        ])
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        if step % 10 == 0:
            gripper_pos = obs['observation'][:3]  # Greifer-Position aus Observation
            distance_to_cube = obs['achieved_goal'][4]  # Greifer-Würfel Distanz
            print(f"  Step {step:2d}: Greifer=[{gripper_pos[0]:.3f}, {gripper_pos[1]:.3f}, {gripper_pos[2]:.3f}], "
                  f"Distanz={distance_to_cube:.3f}, Reward={reward:.2f}")
    
    print("\n PHASE 2: HOVERING SIMULATION (ber dem Würfel stehen bleiben)")
    print("  Simuliere das Problem: Roboter bleibt ber dem Würfel ohne Absenkung")
    
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
    print("   Verstrkte Belohnungen für erfolgreiche Absenkung")

if __name__ == "__main__":
    main()    