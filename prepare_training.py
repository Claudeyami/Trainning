"""
Script chuẩn bị training với environment FIXED
Kiểm tra và tạo các thư mục cần thiết
"""

import os
import shutil
from datetime import datetime


def prepare_training_environment():
    """Chuẩn bị môi trường training"""
    print("🚀 CHUẨN BỊ TRAINING VỚI ENVIRONMENT FIXED")
    print("=" * 50)
    
    # Tạo các thư mục cần thiết
    directories = [
        "model/checkpoints",
        "model/final", 
        "logs",
        "model_backup"
    ]
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"✅ Tạo thư mục: {directory}")
        else:
            print(f"📁 Thư mục đã tồn tại: {directory}")
    
    # Kiểm tra các file cần thiết
    required_files = [
        "g1_standing_env_fixed.py",
        "configs/g1_train_stand_fixed.yaml",
        "train_fixed.py",
        "demo_fixed.py",
        "test_comparison.py"
    ]
    
    print(f"\n📋 KIỂM TRA CÁC FILE CẦN THIẾT:")
    all_files_exist = True
    
    for file in required_files:
        if os.path.exists(file):
            print(f"  ✅ {file}")
        else:
            print(f"  ❌ {file} - THIẾU!")
            all_files_exist = False
    
    # Kiểm tra config file
    config_path = "configs/g1_train_stand_fixed.yaml"
    if os.path.exists(config_path):
        print(f"\n⚙️  CONFIG FILE:")
        with open(config_path, 'r', encoding='utf-8') as f:
            content = f.read()
            if "PD Control gains" in content:
                print(f"  ✅ PD Controller configured")
            if "Assist system" in content:
                print(f"  ✅ Assist system configured")
            if "Governor system" in content:
                print(f"  ✅ Governor system configured")
    
    # Kiểm tra model hiện có
    print(f"\n📁 MODEL HIỆN CÓ:")
    model_files = []
    for root, dirs, files in os.walk("model"):
        for file in files:
            if file.endswith(('.zip', '.pkl')):
                file_path = os.path.join(root, file)
                size = os.path.getsize(file_path) / (1024*1024)
                model_files.append((file, size))
                print(f"  📄 {file} ({size:.1f} MB)")
    
    if not model_files:
        print(f"  📭 Không có model nào")
    
    # Kiểm tra logs
    print(f"\n📊 LOGS:")
    if os.path.exists("logs"):
        log_items = os.listdir("logs")
        if log_items:
            for item in log_items:
                item_path = os.path.join("logs", item)
                if os.path.isdir(item_path):
                    print(f"  📁 {item}/")
                else:
                    size = os.path.getsize(item_path) / (1024*1024)
                    print(f"  📈 {item} ({size:.1f} MB)")
        else:
            print(f"  📭 Không có logs nào")
    
    # Tóm tắt
    print(f"\n🎯 TÓM TẮT:")
    if all_files_exist:
        print(f"  ✅ Tất cả file cần thiết đã sẵn sàng")
        print(f"  ✅ Environment FIXED đã được cài đặt")
        print(f"  ✅ Sẵn sàng cho training")
        
        print(f"\n🚀 CÁC BƯỚC TIẾP THEO:")
        print(f"  1. Test environment: python demo_fixed.py")
        print(f"  2. So sánh với cũ: python test_comparison.py")
        print(f"  3. Training: python train_fixed.py --steps 1000000")
        print(f"  4. Monitor: tensorboard --logdir=logs")
        
    else:
        print(f"  ❌ Thiếu một số file cần thiết")
        print(f"  ❌ Vui lòng kiểm tra lại")
    
    return all_files_exist


def show_training_commands():
    """Hiển thị các lệnh training"""
    print(f"\n💻 CÁC LỆNH TRAINING:")
    print("=" * 30)
    
    commands = [
        ("Test environment", "python demo_fixed.py"),
        ("So sánh environments", "python test_comparison.py"),
        ("Training ngắn (500K steps)", "python train_fixed.py --steps 500000"),
        ("Training đầy đủ (2M steps)", "python train_fixed.py --steps 2000000"),
        ("Training dài (5M steps)", "python train_fixed.py --steps 5000000"),
        ("Monitor training", "tensorboard --logdir=logs"),
        ("Cleanup logs cũ", "python cleanup_old_models.py")
    ]
    
    for desc, cmd in commands:
        print(f"  {desc}:")
        print(f"    {cmd}")
        print()


def main():
    """Main function"""
    print("🎯 CHUẨN BỊ TRAINING VỚI ENVIRONMENT FIXED")
    print("Dựa trên dự án hoạt động tốt với PD Controller + Assist System")
    print("=" * 70)
    
    # Chuẩn bị môi trường
    success = prepare_training_environment()
    
    # Hiển thị lệnh training
    show_training_commands()
    
    if success:
        print(f"🎉 CHUẨN BỊ HOÀN THÀNH!")
        print(f"   - Environment FIXED sẵn sàng")
        print(f"   - Các file cần thiết đã có")
        print(f"   - Có thể bắt đầu training")
    else:
        print(f"⚠️  CHUẨN BỊ CHƯA HOÀN THÀNH!")
        print(f"   - Vui lòng kiểm tra lại các file thiếu")
        print(f"   - Chạy lại script sau khi sửa")


if __name__ == "__main__":
    main()
