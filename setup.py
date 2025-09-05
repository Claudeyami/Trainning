"""
Setup script cho G1 Standing Training Project
"""

import os
import subprocess
import sys


def create_directories():
    """Tạo các thư mục cần thiết"""
    directories = [
        'policies',
        'logs',
        'checkpoints', 
        'best_models',
        'results'
    ]
    
    for dir_name in directories:
        os.makedirs(dir_name, exist_ok=True)
        print(f"✅ Created directory: {dir_name}")


def install_dependencies():
    """Cài đặt dependencies"""
    print("📦 Installing dependencies...")
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])
        print("✅ Dependencies installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False
    return True


def test_environment():
    """Test environment setup"""
    print("🧪 Testing environment setup...")
    try:
        # Test imports
        import torch
        print(f"  ✅ PyTorch {torch.__version__}")
        
        import stable_baselines3
        print(f"  ✅ Stable-Baselines3 {stable_baselines3.__version__}")
        
        import mujoco
        print(f"  ✅ MuJoCo {mujoco.__version__}")
        
        import gymnasium
        print(f"  ✅ Gymnasium {gymnasium.__version__}")
        
        # Test custom environment
        from g1_standing_env import G1StandingEnv
        env = G1StandingEnv()
        obs, info = env.reset()
        env.close()
        print(f"  ✅ G1StandingEnv works (obs shape: {obs.shape})")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Environment test failed: {e}")
        return False


def main():
    """Main setup function"""
    print("🚀 G1 Standing Training Project Setup")
    print("=" * 40)
    
    # Create directories
    create_directories()
    
    # Install dependencies
    if not install_dependencies():
        print("❌ Setup failed at dependency installation")
        return
    
    # Test environment
    if not test_environment():
        print("❌ Setup failed at environment testing")
        return
    
    print("\n✅ Setup completed successfully!")
    print("\n🎯 Next steps:")
    print("1. Kiểm tra config files trong configs/")
    print("2. Chạy training: python run_training.py --check_env --dry_run")
    print("3. Bắt đầu training: python run_training.py")
    print("4. Theo dõi với TensorBoard: tensorboard --logdir ./logs")


if __name__ == "__main__":
    main()
