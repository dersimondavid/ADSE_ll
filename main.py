import gym
import numpy as np
import pybullet as p
import pybullet_data
from gym import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
import matplotlib.pyplot as plt
import time

class StackBlocksEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, gui=True, max_steps=5000, num_blocks=10):
        super().__init__()
        self.gui = gui
        self.max_steps = max_steps
        self.num_blocks = num_blocks
        self.block_size = 0.04

        # Tischparameter
        self.table_base = 0.0                # Tischbeine auf Boden
        self.table_height = 0.65             # Höhe der Tischplatte über Boden

        # Greiferöffnungsbereich [geschlossen, offen]
        self.gripper_opening_range = [0.0, 0.04]

        # Kamera-Parameter für Visualizer
        self.cam_target = [0.5, 0, self.table_height]  # Zielpunkt der Kamera: Tischmitte
        self.cam_distance = 1.2                          # Anfangs-Entfernung
        self.cam_yaw = 50                                # Rotation horizontal
        self.cam_pitch = -35                             # Rotation vertikal

        # PyBullet-Verbindung
        if self.gui:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        # Aktions- und Beobachtungsraum
        self.action_space = spaces.Box(-1, 1, shape=(8,), dtype=np.float32)
        obs_dim = 7 + self.num_blocks * 7 + 1 + 1 + 3 + 3
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)

        # Szene und Blöcke werden beim ersten reset geladen
        self.reset()

    def _setup_scene(self):
        # Plane, Tisch und Markierungen
        self.plane_id = p.loadURDF("plane.urdf")
        # Tisch so platzieren, dass die Beine auf dem Boden stehen
        self.table_id = p.loadURDF(
            "table/table.urdf",
            [0.5, 0, self.table_base],
            useFixedBase=True
        )
        # Markierungen auf der Tischplatte
        top_z = self.table_base + self.table_height
        for start, end in [([0.2, -0.2, top_z], [0.2, 0.2, top_z]),
                           ([0.2, 0.2, top_z], [0.8, 0.2, top_z]),
                           ([0.8, 0.2, top_z], [0.8, -0.2, top_z]),
                           ([0.8, -0.2, top_z], [0.2, -0.2, top_z])]:
            p.addUserDebugLine(start, end, [1, 0, 0], 2)
        # Panda-Roboter auf der Tischplatte
        self.robot = p.loadURDF(
            "franka_panda/panda.urdf",
            [0, 0, top_z],
            useFixedBase=True
        )
        self.arm_joints = list(range(7))     # 7 DoF
        self.gripper_joints = [9, 11]        # Fingerkuppen
        self.ee_link = 11

    def _spawn_blocks(self):
        # zufälliges Platzieren aller Blöcke auf der Tischplatte
        self.block_ids = []
        self.stable_steps = [0] * self.num_blocks
        top_z = self.table_base + self.table_height
        for _ in range(self.num_blocks):
            x = np.random.uniform(0.2, 0.8)
            y = np.random.uniform(-0.2, 0.2)
            z = top_z + self.block_size / 2
            bid = p.loadURDF(
                "cube.urdf",
                [x, y, z],
                globalScaling=self.block_size/0.5
            )
            self.block_ids.append(bid)

    def reset(self):
        # komplette Szene neu initialisieren
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        self._setup_scene()
        self._spawn_blocks()

        # Roboter in Ausgangsstellung
        self.step_count = 0
        for j in self.arm_joints:
            p.resetJointState(self.robot, j, 0.0)
        for j in self.gripper_joints:
            p.resetJointState(self.robot, j, self.gripper_opening_range[1])

        # interne Zustände
        self.prev_base = 0
        self.grasp_constraint = None
        self.held_block = None
        self.lifted = False

        return self._get_obs()

    def step(self, action):
        # Gelenksteuerung
        for i, j in enumerate(self.arm_joints):
            curr = p.getJointState(self.robot, j)[0]
            target = curr + float(action[i]) * 0.03
            p.setJointMotorControl2(self.robot, j,
                                    p.POSITION_CONTROL,
                                    targetPosition=target)
        # Greifersteuerung
        grip_cmd = float(action[7])               # [-1,1]
        frac = (grip_cmd + 1) / 2                 # [0,1]
        target_grip = (self.gripper_opening_range[1]
                      - frac * (self.gripper_opening_range[1]
                                - self.gripper_opening_range[0]))
        for j in self.gripper_joints:
            p.setJointMotorControl2(self.robot, j,
                                    p.POSITION_CONTROL,
                                    targetPosition=target_grip,
                                    force=100)

        # Simulation
        p.stepSimulation()
        if self.gui:
            time.sleep(1./240.)
        self.step_count += 1

        # Beobachtung und Basis-Reward
        obs = self._get_obs()
        base = self._compute_base_reward()
        reward = 0.0

        # Approach-Shaping (skaliert auf [0,1])
        if self.grasp_constraint is None:
            bid = self.block_ids[base]
            pos, _ = p.getBasePositionAndOrientation(bid)
            ee_pos = p.getLinkState(self.robot, self.ee_link)[0]
            d = np.linalg.norm(np.array(ee_pos) - np.array(pos))
            reward += max(0, 1 - d/0.5)
        else:
            cx, cy = 0.5, 0.0
            target_z = (base + 0.5) * self.block_size + self.table_base + self.table_height
            ee_pos = p.getLinkState(self.robot, self.ee_link)[0]
            d2 = np.linalg.norm(np.array(ee_pos)
                                - np.array([cx, cy, target_z]))
            reward += max(0, 1 - d2/0.5)


        # Grasp-Erkennung
        contacts = []
        for link in self.gripper_joints:
            contacts += p.getContactPoints(bodyA=self.robot,
                                           linkIndexA=link)
        holding = any(pt[2] in self.block_ids for pt in contacts)
        if holding and self.grasp_constraint is None and target_grip < 0.02:
            for pt in contacts:
                if pt[2] in self.block_ids:
                    self.held_block = pt[2]
                    break
            self.grasp_constraint = p.createConstraint(
                parentBodyUniqueId=self.robot,
                parentLinkIndex=self.ee_link,
                childBodyUniqueId=self.held_block,
                childLinkIndex=-1,
                jointType=p.JOINT_FIXED,
                jointAxis=[0,0,0],
                parentFramePosition=[0,0,0],
                childFramePosition=[0,0,0]
            )
            reward += 1.0

        # Lift-Bonus
        if self.grasp_constraint is not None and not self.lifted:
            pos, _ = p.getBasePositionAndOrientation(self.held_block)
            if pos[2] > self.block_size * 1.5:
                reward += 1.0
                self.lifted = True

        # Release-Bonus (angepasst auf 1.0)
        if self.grasp_constraint is not None and target_grip > 0.02:
            p.removeConstraint(self.grasp_constraint)
            self.grasp_constraint = None
            new_base = self._compute_base_reward()
            if new_base > self.prev_base:
                reward += 1.0
            self.prev_base = new_base
            self.held_block = None
            self.lifted = False

        # Stabilität & Orientierung normalisiert (Maximalwert jeweils 1)
        stab_raw = self._compute_stability_reward()
        stab_norm = stab_raw / (self.num_blocks * 0.3) if self.num_blocks > 0 else 0
        reward += stab_norm
        ori_raw = self._compute_orientation_reward(base)
        ori_norm = ori_raw / self.num_blocks if self.num_blocks > 0 else 0
        reward += ori_norm

        # Zeitstrafe pro Step
        reward -= 0.01

        # Episode-Ende: Erfolg oder Zeitüberschreitung
        success = (base >= self.num_blocks)
        failure = (self.step_count >= self.max_steps)
        done = success or failure

        # End-Bonus oder Failure-Strafe
        if done:
            if success:
                reward += 10.0
            elif failure:
                reward -= 5.0

        # -------------- Kollisionserkennung --------------
        for pt in p.getContactPoints(bodyA=self.robot, bodyB=self.robot):
            if pt[3] != pt[4]:
                reward -= 5.0
                done = True
                return obs, reward, done, {}

        for pt in p.getContactPoints(bodyA=self.robot):
            other = pt[2]
            link  = pt[3]
            if other in (self.plane_id, self.table_id) and link not in self.gripper_joints:
                reward -= 2.0
                done = True
                return obs, reward, done, {}

        return obs, reward, done, {}

    def _get_obs(self):
        joints = [p.getJointState(self.robot, j)[0]
                  for j in self.arm_joints]
        block_states = []
        for bid in self.block_ids:
            pos, ori = p.getBasePositionAndOrientation(bid)
            block_states += list(pos) + list(ori)   
        stable_flag = float(all(s > 10 for s in self.stable_steps))
        grip_state = np.mean([p.getJointState(self.robot, j)[0]
                              for j in self.gripper_joints])
        base = self._compute_base_reward()
        if base < self.num_blocks:
            target_pos, _ = p.getBasePositionAndOrientation(
                                self.block_ids[base])
        else:
            target_pos = np.array([0.5,0.0,0.5])
        ee_pos = np.array(p.getLinkState(self.robot,
                                         self.ee_link)[0])
        rel_block = list(target_pos - ee_pos)
        top_pos = np.array([0.5,0.0,
                            (base+0.5)*self.block_size])
        rel_top = list(top_pos - ee_pos)
        obs = (joints + block_states +
               [stable_flag, grip_state] +
               rel_block + rel_top)
        return np.array(obs, dtype=np.float32)

    def _compute_base_reward(self):
        tol = 0.02; cx, cy = 0.5, 0.0; count = 0
        for i, bid in enumerate(self.block_ids):
            pos, _ = p.getBasePositionAndOrientation(bid)
            target_z = (i+0.5) * self.block_size
            if (abs(pos[2] - target_z) < tol
                and np.hypot(pos[0]-cx, pos[1]-cy) < tol):
                count += 1
            else:
                break
        return count

    def _compute_stability_reward(self):
        bonus = 0.0
        for idx, bid in enumerate(self.block_ids):
            _, ang_vel = p.getBaseVelocity(bid)
            omega = np.linalg.norm(ang_vel)
            if omega < 0.05:
                self.stable_steps[idx] += 1
                if self.stable_steps[idx] == 10:
                    bonus += 0.3
            else:
                if self.stable_steps[idx] >= 1 and omega > 1.0:
                    bonus -= 0.2
                self.stable_steps[idx] = 0
        return bonus

    def _compute_orientation_reward(self, base):
        bonus = 0.0
        target_quat = np.array([0, 0, 0, 1])
        for i in range(base):
            _, ori = p.getBasePositionAndOrientation(self.block_ids[i])
            ori = np.array(ori)
            dot = abs(np.dot(ori, target_quat))
            ori_err = 1 - dot
            bonus += max(0, 1 - ori_err / 0.1)
        return bonus

    def render(self, mode='human'):
        p.resetDebugVisualizerCamera(self.cam_distance,
                                     self.cam_yaw,
                                     self.cam_pitch,
                                     self.cam_target)

    def close(self):
        p.disconnect()

class RewardPlotCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.ep_rewards = []

    def _on_step(self) -> bool:
        if self.locals.get('dones'):
            self.ep_rewards.append(self.locals['rewards'])
        return True

    def plot(self):
        plt.figure()
        plt.plot(self.ep_rewards)
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title('Learning Curve')
        plt.show()

if __name__ == '__main__':
    env = StackBlocksEnv(gui=True, max_steps=16000, num_blocks=10)
    model = PPO('MlpPolicy', env, verbose=1, learning_rate=1e-3)
    callback = RewardPlotCallback()
    model.learn(total_timesteps=2_000_000, callback=callback)
    model.save('stacking_model_whole_episode.zip')
    callback.plot()
