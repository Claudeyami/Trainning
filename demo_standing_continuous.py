"""
Demo script để chạy model standing LIÊN TỤC với MuJoCo viewer
Sử dụng để test và demo robot standing không ngừng
"""

import os
import numpy as np
import time
from stable_baselines3 import PPO
from g1_standing_simple import G1StandingSimple


def run_continuous_demo(model_path: str, max_episodes: int = 10):
    """
    Chạy demo LIÊN TỤC với model standing đã train
    
    Args:
        model_path: Đường dẫn đến model (.zip)
        max_episodes: Số episode tối đa (0 = chạy vô hạn)
    """
    print(f"🤖 Demo LIÊN TỤC với model: {model_path}")
    print("=" * 60)
    
    # Kiểm tra file tồn tại
    if not os.path.exists(model_path):
        print(f"❌ Model không tồn tại: {model_path}")
        return False
    
    # Kiểm tra VecNormalize file
    vecnormalize_path = model_path.replace('.zip', '_vecnormalize.pkl')
    if not os.path.exists(vecnormalize_path):
        print(f"⚠️  VecNormalize file không tìm thấy: {vecnormalize_path}")
        print("   Model có thể không hoạt động đúng")
    
    try:
        # Load model
        print("📥 Loading PPO model...")
        model = PPO.load(model_path)
        print(f"✅ Model loaded thành công!")
        print(f"   - Policy type: {type(model.policy).__name__}")
        print(f"   - Observation space: {model.observation_space.shape}")
        print(f"   - Action space: {model.action_space.shape}")
        
        # Tạo environment với render
        print("🔧 Tạo environment với MuJoCo viewer...")
        env = G1StandingSimple(
            xml_path="g1_description/g1_23dof.xml",
            config_path="configs/g1_train_stand_fixed.yaml",  # Sử dụng config FIXED
            render_mode="human",  # LUÔN bật render
            max_episode_steps=1000,
            terminate_on_fall=True
        )
        
        # Force render MuJoCo viewer
        print("🖥️  Khởi tạo MuJoCo viewer...")
        try:
            import mujoco
            from mujoco.viewer import launch_passive_viewer
            print("✅ MuJoCo viewer sẵn sàng!")
        except ImportError:
            print("⚠️  MuJoCo viewer không có, sử dụng env.render()")
        except Exception as e:
            print(f"⚠️  MuJoCo viewer lỗi: {e}")
            print("   Sử dụng env.render() thay thế")
        
        # Force render để khởi tạo
        try:
            env.render()
            time.sleep(0.1)
        except:
            pass
        
        # Load VecNormalize nếu có
        if os.path.exists(vecnormalize_path):
            print("📊 Loading VecNormalize...")
            from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
            # Wrap single env thành vector env để load VecNormalize
            vec_env = DummyVecEnv([lambda: env])
            vec_env = VecNormalize.load(vecnormalize_path, vec_env)
            vec_env.training = False  # Không training
            vec_env.norm_reward = False  # Không normalize reward khi test
            print("✅ VecNormalize loaded thành công!")
            
            # Lấy environment gốc từ vector env
            env = vec_env.venv.envs[0]
        else:
            print("⚠️  Không tìm thấy VecNormalize, sử dụng environment gốc")
        
        print(f"✅ Environment khởi tạo thành công!")
        print(f"   - PD Controller: kp={env.kp[:3]}, kd={env.kd[:3]}")
        print(f"   - Action scale: {env.action_scale}")
        print(f"   - Max episodes: {max_episodes if max_episodes > 0 else 'Vô hạn'}")
        
        # Demo loop
        episode = 0
        total_successful_episodes = 0
        total_episodes = 0
        
        print(f"\n🎯 Bắt đầu demo LIÊN TỤC...")
        print("=" * 60)
        print("💡 Nhấn Ctrl+C để dừng demo")
        print("💡 Mỗi episode sẽ tự động reset khi robot ngã")
        print("=" * 60)
        
        try:
            while max_episodes == 0 or episode < max_episodes:
                episode += 1
                total_episodes += 1
                
                print(f"\n🔄 Episode {episode}")
                print("-" * 40)
                
                # Reset environment
                obs, info = env.reset()
                print(f"🌍 Initial state:")
                print(f"   - Height: {info['height']:.3f}m")
                print(f"   - Uprightness: {info['uprightness']:.3f}")
                print(f"   - PD Controller: kp={env.kp[:3]}, kd={env.kd[:3]}")
                print(f"   - Action scale: {env.action_scale}")
                
                # Episode loop
                step = 0
                total_reward = 0.0
                max_standing_duration = 0.0
                successful_standing = False
                
                while True:
                    # Predict action
                    action, _ = model.predict(obs, deterministic=True)
                    
                    # Step environment
                    obs, reward, terminated, truncated, info = env.step(action)
                    
                    total_reward += reward
                    step += 1
                    
                    # Track standing duration
                    current_height = info.get('height', 0.0)
                    current_uprightness = info.get('uprightness', 0.0)
                    max_standing_duration = max(max_standing_duration, step * 0.01)  # Convert steps to seconds
                    
                    # Check success (height > 0.7m and uprightness > 0.9)
                    if current_height >= 0.7 and current_uprightness >= 0.9 and not successful_standing:
                        successful_standing = True
                        print(f"🎉 SUCCESS! Robot đạt height={current_height:.2f}m, uprightness={current_uprightness:.2f}")
                    
                    # Print progress every 2 seconds
                    if step % 100 == 0:
                        print(f"⏱️  Step {step}: height={current_height:.3f}m, "
                              f"uprightness={current_uprightness:.3f}, reward={reward:.1f}")
                    
                    # Check termination
                    if terminated:
                        print(f"⛔ Episode {episode} terminated: {info.get('termination_reason', 'unknown')}")
                        break
                    
                    if truncated:
                        print(f"⏰ Episode {episode} truncated: timeout")
                        break
                    
                    # Small delay để render mượt
                    time.sleep(0.05)  # Tăng delay để coi rõ hơn
                    
                    # Force render mỗi step để đảm bảo UI hiển thị
                    try:
                        env.render()
                    except:
                        pass  # Bỏ qua nếu render lỗi
                
                # Episode summary
                print(f"📊 Episode {episode} Summary:")
                print(f"   - Steps: {step}")
                print(f"   - Total reward: {total_reward:.1f}")
                print(f"   - Duration: {max_standing_duration:.2f}s")
                print(f"   - Final height: {info['height']:.3f}m")
                print(f"   - Final uprightness: {info['uprightness']:.3f}")
                
                if successful_standing:
                    print(f"   ✅ SUCCESS: Robot đạt mục tiêu!")
                    total_successful_episodes += 1
                else:
                    print(f"   ❌ FAILED: Robot chưa đạt mục tiêu (height < 0.7m hoặc uprightness < 0.9)")
                
                # Success rate
                success_rate = (total_successful_episodes / total_episodes) * 100
                print(f"📈 Success rate: {total_successful_episodes}/{total_episodes} ({success_rate:.1f}%)")
                
                # Auto-reset cho episode tiếp theo
                print(f"🔄 Tự động reset cho episode tiếp theo...")
                time.sleep(1.0)  # Đợi 1 giây để xem kết quả cuối
                
        except KeyboardInterrupt:
            print(f"\n⏹️  Demo bị dừng bởi user (Ctrl+C)")
        
        # Final summary
        print(f"\n🎯 Final Demo Summary")
        print("=" * 60)
        print(f"   - Total episodes: {total_episodes}")
        print(f"   - Successful episodes: {total_successful_episodes}")
        print(f"   - Success rate: {(total_successful_episodes/total_episodes*100):.1f}%")
        
        if total_successful_episodes > 0:
            print(f"   ✅ Robot đã học được standing!")
        else:
            print(f"   ❌ Robot chưa học được standing, cần training thêm")
        
        # Close environment
        print("🖥️  Giữ UI mở thêm 5 giây để coi...")
        time.sleep(5.0)  # Giữ UI mở thêm 5 giây
        env.close()
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def list_available_models():
    """Liệt kê các model có sẵn trong thư mục model"""
    print("📁 Available models in model/ directory:")
    print("=" * 50)
    
    # Tìm tất cả models trong model/ và subdirectories
    model_files = []
    model_paths = []
    
    # Tìm trong model/ (root)
    if os.path.exists("model"):
        for f in os.listdir("model"):
            if f.endswith(".zip"):
                model_files.append(f)
                model_paths.append(os.path.join("model", f))
    
    # Tìm trong policies/final/ (backup)
    final_dir = "policies/final"
    if os.path.exists(final_dir):
        for f in os.listdir(final_dir):
            if f.endswith(".zip") and f not in model_files:
                model_files.append(f)
                model_paths.append(os.path.join(final_dir, f))
    
    # Tìm trong policies/checkpoints/ (backup)
    checkpoints_dir = "policies/checkpoints"
    if os.path.exists(checkpoints_dir):
        for f in os.listdir(checkpoints_dir):
            if f.endswith(".zip") and f not in model_files:
                model_files.append(f)
                model_paths.append(os.path.join(checkpoints_dir, f))
    
    if not model_files:
        print("❌ Không tìm thấy model nào (.zip files)")
        return []
    
    for i, (model_file, model_path) in enumerate(zip(model_files, model_paths)):
        file_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
        
        print(f"   {i+1}. {model_file}")
        print(f"      Path: {model_path}")
        print(f"      Size: {file_size:.1f} MB")
        
        # Kiểm tra VecNormalize
        vecnormalize_path = model_path.replace('.zip', '_vecnormalize.pkl')
        if os.path.exists(vecnormalize_path):
            print(f"      ✅ VecNormalize: Có")
        else:
            print(f"      ❌ VecNormalize: Không")
    
    return model_paths


def main():
    """Main function"""
    print("🤖 G1 Standing Continuous Demo - GIAI ĐOẠN 1")
    print("=" * 60)
    print("🎯 Demo LIÊN TỤC với MuJoCo viewer")
    print("💡 Robot sẽ tự động reset khi ngã")
    print("💡 Nhấn Ctrl+C để dừng demo")
    print("=" * 60)
    
    # Liệt kê models có sẵn
    available_models = list_available_models()
    
    if not available_models:
        print("\n💡 Để train model mới:")
        print("   python train_simple.py")
        return
    
    # Chọn model để chạy
    print(f"\n🎯 Chọn model để demo (1-{len(available_models)}):")
    
    try:
        choice = int(input("   Enter choice: ")) - 1
        if 0 <= choice < len(available_models):
            model_path = available_models[choice]
            selected_model = os.path.basename(model_path)
            
            print(f"\n✅ Đã chọn: {selected_model}")
            print(f"   Path: {model_path}")
            
            # Hỏi số episode
            try:
                episodes_input = input("   Số episode (0 = vô hạn): ")
                max_episodes = int(episodes_input) if episodes_input.strip() != "0" else 0
            except ValueError:
                max_episodes = 5  # Default 5 episodes
            
            if max_episodes == 0:
                print("   🚀 Demo sẽ chạy VÔ HẠN (nhấn Ctrl+C để dừng)")
            else:
                print(f"   🚀 Demo sẽ chạy {max_episodes} episodes")
            
            # Chạy demo
            print(f"\n🚀 Bắt đầu demo LIÊN TỤC...")
            success = run_continuous_demo(model_path, max_episodes)
            
            if success:
                print(f"\n🎉 Demo hoàn thành!")
            else:
                print(f"\n❌ Demo thất bại!")
                
        else:
            print("❌ Lựa chọn không hợp lệ!")
            
    except (ValueError, KeyboardInterrupt):
        print("\n⏹️  Hủy bỏ")


if __name__ == "__main__":
    main()
