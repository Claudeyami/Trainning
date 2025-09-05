"""
G1 Standing Environment - VERSION ĐƠN GIẢN
Chỉ để robot đứng thẳng, không nhảy lên cao
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco
import yaml
import os
import math
from typing import Dict, Any, Tuple, Optional


class G1StandingSimple(gym.Env):
    """
    G1 Standing Environment - ĐƠN GIẢN
    Chỉ để robot đứng thẳng ở độ cao thấp
    """
    
    def __init__(
        self,
        xml_path: str = "g1_description/g1_23dof.xml",
        config_path: str = "configs/g1_train_stand_fixed.yaml",
        render_mode: Optional[str] = None,
        max_episode_steps: int = 1000,
        terminate_on_fall: bool = True,
    ):
        super().__init__()
        
        # Load config
        self.xml_path = xml_path
        self.config_path = config_path
        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps
        self.terminate_on_fall = terminate_on_fall
        self.current_step = 0
        
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # MuJoCo setup
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.viewer = None
        
        # Thiết lập trọng lực và physics
        self.model.opt.gravity[2] = -9.81
        self.model.opt.timestep = 0.002
        self.model.opt.integrator = 0  # Euler
        
        # Control parameters
        self.dt = 0.002
        self.decim = 5
        self.action_scale = 0.05  # VỪA PHẢI để đứng tự nhiên
        
        # PD Control gains - MẠNH HỚN để giữ thăng bằng
        self.default = np.array([-0.12, 0.00, +0.08, +0.34, -0.16, +0.02, -0.12, 0.00, +0.08, +0.34, -0.16, +0.02], dtype=np.float32)
        self.kp = np.array([150, 120, 150, 240, 150, 75, 150, 120, 150, 240, 150, 75], dtype=np.float32)
        self.kd = np.array([8, 8, 8, 12, 8, 5, 8, 8, 8, 12, 8, 5], dtype=np.float32)
        
        # Action và observation space
        self.num_actions = 12
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.num_actions,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(47,), dtype=np.float32)
        
        # Actuator limits
        self.ctrl_min = None
        self.ctrl_max = None
        
        # Base body ID
        try:
            self.base_bid = self.model.body("pelvis").id
        except:
            self.base_bid = 0
        
        # Lấy 12 joints đầu tiên
        self.actuated_joint_ids = []
        for i in range(min(self.model.nu, self.num_actions)):
            trn_type = self.model.actuator_trntype[i]
            if trn_type == mujoco.mjtTrn.mjTRN_JOINT:
                j_id = int(self.model.actuator_trnid[i, 0])
                self.actuated_joint_ids.append(j_id)
        
        # State tracking
        self.prev_action = np.zeros(self.num_actions, dtype=np.float32)
        
        # Assist system - HỖ TRỢ ĐỂ ĐỨNG VỮNG
        self.assist_enabled = True
        self.assist_strength = 0.8  # Độ mạnh của assist - TĂNG MẠNH
        self.assist_decay = 0.995  # Giảm dần theo thời gian - CHẬM HỚN
        
        # Governor system - KIỂM SOÁT HÀNH ĐỘNG
        self.governor_enabled = True
        self.max_base_velocity = 2.0  # Tốc độ tối đa của base
        self.max_pitch_angle = 0.3  # Góc nghiêng tối đa
        
        print(f"🎯 G1StandingSimple - ĐỨNG THẲNG!")
        print(f"  - Action scale: {self.action_scale}")
        print(f"  - PD gains: kp={self.kp[:3]}, kd={self.kd[:3]}")
    
    def _get_observation(self):
        """Tạo observation vector"""
        # Lấy joint positions và velocities
        q = []
        dq = []
        for joint_id in self.actuated_joint_ids:
            qpos_idx = self.model.jnt_qposadr[joint_id]
            qvel_idx = self.model.jnt_dofadr[joint_id]
            q.append(self.data.qpos[qpos_idx])
            dq.append(self.data.qvel[qvel_idx])
        
        q = np.array(q, dtype=np.float32)
        dq = np.array(dq, dtype=np.float32)
        omega = self.data.qvel[3:6].copy()
        quat = self.data.qpos[3:7].copy()
        
        # Phase for standing
        phase = 0.0
        ph = np.array([np.sin(phase), np.cos(phase)], dtype=np.float32)
        
        return np.concatenate([
            omega * 0.25,
            self._gravity_orientation(quat),
            np.array([0.0, 0.0, 0.0]),  # cmd
            (q - self.default) * 1.0,
            dq * 0.05,
            np.zeros(self.num_actions),  # prev_action
            ph
        ], axis=0).astype(np.float32)
    
    def _gravity_orientation(self, quat):
        """Convert quaternion to gravity vector"""
        qw, qx, qy, qz = quat
        g0 = 2 * (-qz * qx + qw * qy)
        g1 = -2 * (qz * qy + qw * qx)
        g2 = 1 - 2 * (qx*qx + qy*qy)
        return np.array([g0, g1, g2], dtype=np.float32)
    
    def _pd_control(self, q, dq, q_ref):
        """PD Controller"""
        tau = self.kp * (q_ref - q) + self.kd * (0.0 - dq)
        return tau
    
    def _compute_assist_torque(self, q, dq):
        """Assist system - Hỗ trợ để đứng vững"""
        assist_tau = np.zeros(self.num_actions, dtype=np.float32)
        
        if len(q) >= 12:
            # Assist cho chân để giữ thăng bằng - MẠNH HỚN
            for i in range(6):  # 6 joints chân trái
                if i < len(q):
                    # Assist về vị trí mặc định
                    error = self.default[i] - q[i]
                    assist_tau[i] = 0.3 * error  # Assist MẠNH HỚN
            
            for i in range(6, 12):  # 6 joints chân phải
                if i < len(q):
                    # Assist về vị trí mặc định
                    error = self.default[i] - q[i]
                    assist_tau[i] = 0.3 * error  # Assist MẠNH HỚN
            
            # Assist cho việc giữ thăng bằng - THÊM MỚI
            if len(q) >= 6:
                # Assist cho base để giữ thẳng
                base_roll = q[3] if len(q) > 3 else 0
                base_pitch = q[4] if len(q) > 4 else 0
                assist_tau[3] += -0.5 * base_roll  # Giảm roll
                assist_tau[4] += -0.5 * base_pitch  # Giảm pitch
        
        return assist_tau
    
    def _apply_governor(self, tau, q, dq):
        """Governor system - Kiểm soát hành động"""
        # Giới hạn tốc độ base
        if len(dq) >= 6:
            base_velocity = np.linalg.norm(dq[:3])  # Tốc độ linear
            if base_velocity > self.max_base_velocity:
                scale = self.max_base_velocity / base_velocity
                tau[:6] *= scale
        
        # Giới hạn góc nghiêng
        if len(q) >= 6:
            pitch_angle = abs(q[4])  # Góc pitch
            if pitch_angle > self.max_pitch_angle:
                # Giảm torque nếu nghiêng quá mức
                scale = self.max_pitch_angle / pitch_angle
                tau *= scale
        
        return tau
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict]:
        """Reset environment"""
        super().reset(seed=seed)
        
        self.current_step = 0
        mujoco.mj_resetData(self.model, self.data)
        
        # Set initial pose
        for i, joint_id in enumerate(self.actuated_joint_ids):
            qpos_idx = self.model.jnt_qposadr[joint_id]
            if i < len(self.default):
                self.data.qpos[qpos_idx] = self.default[i]
        
        # Set height - THEO CHIỀU CAO THỰC TẾ CỦA ROBOT (0.793m)
        self.data.qpos[2] = 0.793
        
        # Set upright orientation
        self.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        
        # Add small noise
        noise_scale = 0.005
        qpos_noise = self.np_random.normal(0, noise_scale, size=self.data.qpos.shape)
        qvel_noise = self.np_random.normal(0, noise_scale * 0.1, size=self.data.qvel.shape)
        qpos_noise[3:7] = 0.0
        self.data.qpos += qpos_noise
        self.data.qvel += qvel_noise
        
        mujoco.mj_forward(self.model, self.data)
        
        obs = self._get_observation()
        up_z = float(self.data.xmat[self.base_bid, 8]) if self.base_bid >= 0 else 1.0
        info = {
            'height': self.data.qpos[2],
            'uprightness': up_z,
            'gravity': self.model.opt.gravity[2]
        }
        
        return obs, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Step environment"""
        self.current_step += 1
        action = np.clip(action, -1.0, 1.0).astype(np.float32)
        prev = self.prev_action.copy()
        self.prev_action = action.copy()
        
        reward = 0.0
        terminated = False
        info = {}
        
        # Simulation step
        for _ in range(self.decim):
            # Lấy joint positions và velocities
            q = []
            dq = []
            for joint_id in self.actuated_joint_ids:
                qpos_idx = self.model.jnt_qposadr[joint_id]
                qvel_idx = self.model.jnt_dofadr[joint_id]
                q.append(self.data.qpos[qpos_idx])
                dq.append(self.data.qvel[qvel_idx])
            
            q = np.array(q, dtype=np.float32)
            dq = np.array(dq, dtype=np.float32)
            
            # PD Control - THEO DỰ ÁN HOÀN THÀNH
            target_q = self.default + action * self.action_scale
            tau = self._pd_control(q, dq, target_q)
            
            # ASSIST SYSTEM - HỖ TRỢ ĐỂ ĐỨNG VỮNG
            if self.assist_enabled:
                # Assist cho việc giữ thăng bằng
                assist_tau = self._compute_assist_torque(q, dq)
                tau += assist_tau * self.assist_strength
                # Giảm assist theo thời gian
                self.assist_strength *= self.assist_decay
            
            # GOVERNOR SYSTEM - KIỂM SOÁT HÀNH ĐỘNG
            if self.governor_enabled:
                tau = self._apply_governor(tau, q, dq)
            
            self.data.ctrl[:self.num_actions] = tau
            
            mujoco.mj_step(self.model, self.data)
            
            # Reward function - THEO DỰ ÁN HOÀN THÀNH
            current_height = self.data.qpos[2]
            up_z = float(self.data.xmat[self.base_bid, 8]) if self.base_bid >= 0 else 1.0
            
            # Height reward - THEO CHIỀU CAO THỰC TẾ CỦA ROBOT (0.793m)
            target_height = 0.793
            height_reward = np.exp(-((current_height - target_height)**2) / (2 * 0.05**2))
            
            # Upright reward - MẠNH HỚN để giữ thăng bằng
            upright_reward = max(0.0, up_z) ** 2  # Bình phương để tăng cường
            
            # Stability reward - Thưởng cho việc giữ thăng bằng
            stability_reward = 0.0
            if len(q) >= 12:
                # Penalty cho việc nghiêng quá mức
                base_roll = abs(q[3]) if len(q) > 3 else 0  # base roll
                base_pitch = abs(q[4]) if len(q) > 4 else 0  # base pitch
                stability_reward = -0.5 * (base_roll + base_pitch)
            
            # Energy penalty (nhẹ)
            energy_penalty = -1e-4 * float((tau**2).mean())
            
            # Smooth penalty (nhẹ)
            smooth_penalty = -1e-3 * float(((action - prev)**2).mean())
            
            # PENALTY MẠNH cho việc nhảy lên cao
            if current_height > 1.0:  # Nếu cao hơn 1m (quá cao so với 0.793m)
                height_penalty = -50.0 * (current_height - 1.0)  # Penalty mạnh
            else:
                height_penalty = 0.0
            
            # PENALTY cho tư thế không tự nhiên (tay dang rộng, chân nhấc lên)
            # Penalty cho việc tay dang rộng quá mức
            arm_penalty = 0.0
            if len(q) >= 12:  # Đảm bảo có đủ joint angles
                # Left arm (shoulder joints)
                left_shoulder_roll = abs(q[6]) if len(q) > 6 else 0  # left_shoulder_roll
                left_shoulder_pitch = abs(q[7]) if len(q) > 7 else 0  # left_shoulder_pitch
                # Right arm (shoulder joints)  
                right_shoulder_roll = abs(q[12]) if len(q) > 12 else 0  # right_shoulder_roll
                right_shoulder_pitch = abs(q[13]) if len(q) > 13 else 0  # right_shoulder_pitch
                
                # Penalty nếu tay dang quá rộng
                arm_penalty = -0.1 * (left_shoulder_roll + left_shoulder_pitch + right_shoulder_roll + right_shoulder_pitch)
            
            reward += 0.6 * height_reward + 1.0 * upright_reward + stability_reward + energy_penalty + smooth_penalty + height_penalty + arm_penalty
            
            # Termination check - THEO DỰ ÁN HOÀN THÀNH
            if self.terminate_on_fall and (current_height < 0.20 or up_z < 0.3):
                terminated = True
                break
        
        # Check timeout
        truncated = self.current_step >= self.max_episode_steps
        
        # Get observation
        obs = self._get_observation()
        
        # Info
        info = {
            "height": self.data.qpos[2],
            "uprightness": up_z if self.base_bid >= 0 else 1.0,
            "total_reward": reward
        }
        
        return obs, float(reward), terminated, truncated, info
    
    def render(self):
        """Render environment"""
        if self.render_mode == "human":
            try:
                if self.viewer is None:
                    import mujoco.viewer
                    self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
                self.viewer.sync()
            except AttributeError:
                pass
    
    def close(self):
        """Close environment"""
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None


if __name__ == "__main__":
    """Test environment"""
    print("🧪 Testing G1StandingSimple...")
    
    env = G1StandingSimple()
    
    # Test một episode
    obs, info = env.reset()
    print(f"✅ Environment reset thành công!")
    print(f"   - Observation shape: {obs.shape}")
    print(f"   - Action shape: {env.action_space.shape}")
    
    total_reward = 0
    for step in range(100):
        action = env.action_space.sample() * 0.1  # Small actions
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        
        if step % 20 == 0:
            print(f"Step {step}: reward={reward:.3f}, height={info['height']:.3f}, "
                  f"uprightness={info['uprightness']:.3f}")
        
        if done or truncated:
            break
    
    print(f"✅ Test completed!")
    print(f"   - Total reward: {total_reward:.3f}")
    print(f"   - Final height: {info['height']:.3f}")
    print(f"   - Final uprightness: {info['uprightness']:.3f}")
    
    env.close()
