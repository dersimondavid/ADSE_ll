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

        # PyBullet initialisieren
        if self.gui:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)

        # --- Action: 7 Gelenke + Greiferöffnung
        self.action_space = spaces.Box(-1, 1, shape=(8,), dtype=np.float32)

        # --- Observation:
        # 7 Gelenkwinkel
        # + für jeden Block: pos(3)+ori(4)
        # + ein Stabilitätsflag
        # + Greiferöffnung (1)
        # + Relativer Vektor zum Zielblock (3)
        # + Relativer Vektor zum Turm-Top (3)
        obs_dim = 7 + self.num_blocks * 7 + 1 + 1 + 3 + 3
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)

        # Szene und Reset
        self._setup_scene()
        self.reset()

    def _setup_scene(self):
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        p.loadURDF("plane.urdf")
        p.loadURDF("table/table.urdf", [0.5, 0, -0.65], useFixedBase=True)
        # roter Rahmen für Stapelbereich
        for start, end in [([0.2, -0.2, 0], [0.2, 0.2, 0]),
                           ([0.2, 0.2, 0], [0.8, 0.2, 0]),
                           ([0.8, 0.2, 0], [0.8, -0.2, 0]),
                           ([0.8, -0.2, 0], [0.2, -0.2, 0])]:
            p.addUserDebugLine(start, end, [1, 0, 0], 2)
        # Panda-Roboter laden
        self.robot = p.loadURDF("franka_panda/panda.urdf", [0, 0, 0], useFixedBase=True)
        self.arm_joints = list(range(7))
        self.gripper_joints = [9, 11]
        self.ee_link = 11

    def reset(self):
        self.step_count = 0
        # Roboter in Nullstellung
        for j in self.arm_joints:
            p.resetJointState(self.robot, j, 0.0)
        for j in self.gripper_joints:
            p.resetJointState(self.robot, j, 0.04)

        # alte Blöcke entfernen und neu platzieren
        for bid in getattr(self, 'block_ids', []):
            p.removeBody(bid)
        self.block_ids = []
        for _ in range(self.num_blocks):
            x = np.random.uniform(0.2, 0.8)
            y = np.random.uniform(-0.2, 0.2)
            z = self.block_size / 2
            bid = p.loadURDF("cube.urdf",
                             [x, y, z],
                             globalScaling=self.block_size/0.5)
            self.block_ids.append(bid)

        # interne Flags für Reward-Phasen
        self.prev_base = 0
        self.grasp_constraint = None
        self.held_block = None
        self.grasped_last_step = False
        self.lifted = False

        return self._get_obs()

    def step(self, action):
        # 1) Gelenksteuerung
        for i, j in enumerate(self.arm_joints):
            curr = p.getJointState(self.robot, j)[0]
            target = curr + float(action[i]) * 0.05
            p.setJointMotorControl2(self.robot, j,
                                    p.POSITION_CONTROL, target)
        grip = float(action[7])
        for j in self.gripper_joints:
            p.setJointMotorControl2(self.robot, j,
                                    p.POSITION_CONTROL, grip)

        # Simulation
        p.stepSimulation()
        if self.gui:
            time.sleep(1. / 240.)
        self.step_count += 1

        # 2) Beobachtung und Basis-Werte
        obs = self._get_obs()
        base = self._compute_base_reward()

        # 3) Belohnungskomponenten
        reward = 0.0

        # — Approach-Shaping
        if self.grasp_constraint is None:
            # Abstand zum nächsten Block
            bid = self.block_ids[base]
            pos, _ = p.getBasePositionAndOrientation(bid)
            ee_pos = p.getLinkState(self.robot, self.ee_link)[0]
            d = np.linalg.norm(np.array(ee_pos) - np.array(pos))
            reward += 0.5 * max(0, 1 - d/0.5)
        else:
            # Abstand zum Stack-Top
            cx, cy = 0.5, 0.0
            target_z = (base + 0.5) * self.block_size
            ee_pos = p.getLinkState(self.robot, self.ee_link)[0]
            d2 = np.linalg.norm(np.array(ee_pos) - np.array([cx, cy, target_z]))
            reward += 0.5 * max(0, 1 - d2/0.5)

        # — Grasp-Bonus (erstes Fixieren)
        contacts = p.getContactPoints(bodyA=self.robot, linkIndexA=self.ee_link)
        holding = any(pt[2] in self.block_ids for pt in contacts)
        if holding and self.grasp_constraint is None and grip < 0.02:
            # Constraint erzeugen
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
                jointAxis=[0, 0, 0],
                parentFramePosition=[0, 0, 0],
                childFramePosition=[0, 0, 0]
            )
            reward += 1.0  # Grasp-Bonus
            self.grasped_last_step = True

        # — Lift-Bonus
        if self.grasp_constraint is not None and not self.lifted:
            pos, _ = p.getBasePositionAndOrientation(self.held_block)
            if pos[2] > self.block_size * 1.5:
                reward += 1.0
                self.lifted = True

        # — Release-Bonus 
        if self.grasp_constraint is not None and grip > 0.02:
            p.removeConstraint(self.grasp_constraint)
            self.grasp_constraint = None
            # wenn korrekt abgelegt (base > prev_base)
            placement_bonus = 0.0
            new_base = self._compute_base_reward()
            if new_base > self.prev_base:
                placement_bonus = 2.0
            reward += placement_bonus
            self.prev_base = new_base
            # Flags zurücksetzen
            self.held_block = None
            self.grasped_last_step = False
            self.lifted = False

        # — Stability & Orientation
        reward += self._compute_stability_reward() * 0.3
        reward += self._compute_orientation_reward(base) * 1.0

        # 4) Done-Flag
        time_up = (self.step_count >= self.max_steps) and (self.grasp_constraint is None)
        done = (base >= self.num_blocks) or time_up

        return obs, reward, done, {}

    def _get_obs(self):
        # Gelenkwinkel
        joints = [p.getJointState(self.robot, j)[0] for j in self.arm_joints]
        # Blöcke pos+ori
        block_states = []
        for bid in self.block_ids:
            pos, ori = p.getBasePositionAndOrientation(bid)
            block_states += list(pos) + list(ori)
        # Stabilitätsflag
        stable_flag = float(all(s > 10 for s in getattr(self, 'stable_steps', [0]*self.num_blocks)))
        # Greiferöffnung (Mittelwert der beiden Seiten)
        grip_state = np.mean([p.getJointState(self.robot, j)[0] for j in self.gripper_joints])
        # Relative Vektoren
        base = self._compute_base_reward()
        if base < self.num_blocks:
            target_pos, _ = p.getBasePositionAndOrientation(self.block_ids[base])
        else:
            target_pos = np.array([0.5, 0.0, 0.5])
        ee_pos = np.array(p.getLinkState(self.robot, self.ee_link)[0])
        rel_block = list(target_pos - ee_pos)
        # Turm-Top:
        top_pos = np.array([0.5, 0.0, (base+0.5)*self.block_size])
        rel_top = list(top_pos - ee_pos)

        obs = joints + block_states + [stable_flag, grip_state] + rel_block + rel_top
        return np.array(obs, dtype=np.float32)

    def _compute_base_reward(self):
        tol = 0.02
        cx, cy = 0.5, 0.0
        count = 0
        for i, bid in enumerate(self.block_ids):
            pos, _ = p.getBasePositionAndOrientation(bid)
            target_z = (i + 0.5) * self.block_size
            if abs(pos[2] - target_z) < tol and np.hypot(pos[0]-cx, pos[1]-cy) < tol:
                count += 1
            else:
                break
        return count

    def _compute_stability_reward(self):
        bonus = 0.0
        if not hasattr(self, 'stable_steps'):
            self.stable_steps = [0] * self.num_blocks
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
        pass

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
    env = StackBlocksEnv(gui=True, max_steps=5000, num_blocks=10)
    model = PPO('MlpPolicy', env, verbose=1, learning_rate=1e-3)
    callback = RewardPlotCallback()
    model.learn(total_timesteps=500_000, callback=callback)
    model.save('stacking_model_with_grasp.zip')
    callback.plot()
