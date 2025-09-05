"""
Training script đơn giản cho G1StandingSimple
"""

import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from g1_standing_simple import G1StandingSimple


def make_env():
    """Tạo environment"""
    return G1StandingSimple(
        xml_path="g1_description/g1_23dof.xml",
        config_path="configs/g1_train_stand_fixed.yaml",
        render_mode=None,
        max_episode_steps=1000,
        terminate_on_fall=True
    )


def main():
    """Main training function"""
    print("🤖 G1 Standing Training - ĐƠN GIẢN")
    print("=" * 50)
    
    # Tạo thư mục
    os.makedirs("model", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    # Tạo environment
    print("🔧 Tạo environment...")
    env = DummyVecEnv([make_env])
    env = VecNormalize(env, training=True, norm_obs=True, norm_reward=True, clip_obs=10.0)
    
    # Tạo model
    print("🧠 Tạo PPO model...")
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        tensorboard_log="./logs",
        device="auto"
    )
    
    print("🚀 Bắt đầu training...")
    
    try:
        model.learn(
            total_timesteps=1000000,  # 1M steps
            progress_bar=True,
            tb_log_name="g1_standing"
        )
        
        print("✅ Training hoàn thành!")
        
        # Save model
        model_path = "model/g1_standing_final"
        model.save(model_path)
        env.save(f"{model_path}_vecnormalize.pkl")
        
        print(f"💾 Model saved: {model_path}.zip")
        
    except KeyboardInterrupt:
        print("⏹️  Training bị dừng")
        model_path = "model/g1_standing_interrupted"
        model.save(model_path)
        env.save(f"{model_path}_vecnormalize.pkl")
        print(f"💾 Model saved: {model_path}.zip")
    
    finally:
        env.close()


if __name__ == "__main__":
    main()
