# G1 Robot Standing Training Project

## 📋 Tổng quan dự án

Dự án này tập trung vào việc huấn luyện robot Unitree G1 học cách đứng thẳng và giữ thăng bằng sử dụng Reinforcement Learning (RL) với môi trường MuJoCo. Dự án sử dụng thuật toán PPO (Proximal Policy Optimization) từ thư viện Stable Baselines3.

## 🤖 Về Robot G1

Unitree G1 là một humanoid robot cao cấp với:
- **Chiều cao**: 0.793m (theo thiết kế thực tế)
- **Số bậc tự do**: 23 DOF (sử dụng model `g1_23dof`)
- **Cấu trúc**: 2 chân (6 DOF mỗi chân) + 2 tay (5 DOF mỗi tay) + 1 thân (1 DOF)
- **Mục tiêu**: Học cách đứng thẳng và giữ thăng bằng ổn định

## 🎯 Mục tiêu dự án

- **Chính**: Robot học cách đứng thẳng ở độ cao 0.793m (chiều cao thực tế)
- **Phụ**: Giữ thăng bằng ổn định, không ngã, không nhảy lên cao
- **Kỹ thuật**: Sử dụng PD Controller + Assist System + Governor System

## 🏗️ Cấu trúc dự án

```
Train/
├── 📁 configs/                    # Cấu hình training
│   └── g1_train_stand_fixed.yaml  # Config chính cho standing
├── 📁 g1_description/             # Mô hình robot G1
│   ├── g1_23dof.xml              # Model 23 DOF (chính)
│   ├── g1_29dof.xml              # Model 29 DOF
│   ├── meshes/                    # File 3D mesh
│   └── README.md                  # Thông tin chi tiết về model
├── 📁 model/                      # Model đã train
│   ├── g1_standing_final.zip     # Model PPO chính
│   └── g1_standing_final_vecnormalize.pkl  # VecNormalize
├── 📁 logs/                       # Log training
├── 📁 policies/                   # Backup models
├── 🤖 g1_standing_simple.py      # Environment chính
├── 🚀 train.py                    # Script training
├── 🎮 demo_standing_continuous.py # Demo liên tục
├── 📦 requirements.txt            # Dependencies
└── 📖 README.md                   # File này
```

## 🚀 Cài đặt

### 1. Cài đặt dependencies

```bash
pip install -r requirements.txt
```

### 2. Kiểm tra MuJoCo

```bash
python -c "import mujoco; print('MuJoCo OK')"
```

## 🎮 Sử dụng

### 1. Training model mới

```bash
python train.py
```

**Thông số training:**
- **Algorithm**: PPO (Proximal Policy Optimization)
- **Timesteps**: 1,000,000 steps
- **Learning rate**: 3e-4
- **Batch size**: 64
- **N epochs**: 10
- **Gamma**: 0.99

### 2. Demo model đã train

```bash
python demo_standing_continuous.py
```

**Tính năng demo:**
- ✅ Chạy liên tục với MuJoCo viewer
- ✅ Tự động reset khi robot ngã
- ✅ Hiển thị thống kê real-time
- ✅ Có thể chọn số episode (0 = vô hạn)

### 3. Test environment

```bash
python g1_standing_simple.py
```

## 🔧 Cấu hình kỹ thuật

### PD Controller
```yaml
kps: [150, 120, 150, 240, 150, 75, 150, 120, 150, 240, 150, 75]
kds: [8, 8, 8, 12, 8, 5, 8, 8, 8, 12, 8, 5]
```

### Action Space
- **Dimension**: 12 (6 joints mỗi chân)
- **Range**: [-1.0, 1.0]
- **Scale**: 0.05 (vừa phải để đứng tự nhiên)

### Observation Space
- **Dimension**: 47
- **Components**:
  - Angular velocity (3) × 0.25
  - Gravity orientation (3)
  - Command (3) - luôn [0,0,0] cho standing
  - Joint positions (12) - relative to default
  - Joint velocities (12) × 0.05
  - Previous action (12)
  - Phase (2) - sin/cos cho standing

### Reward Function
```python
reward = 0.6 * height_reward + 1.0 * upright_reward + stability_reward + 
         energy_penalty + smooth_penalty + height_penalty + arm_penalty
```

**Chi tiết:**
- **Height reward**: Gaussian centered at 0.793m (chiều cao thực tế)
- **Upright reward**: Bình phương của up_z (mạnh hơn)
- **Stability reward**: Penalty cho roll/pitch quá mức
- **Height penalty**: Penalty mạnh nếu cao > 1.0m
- **Arm penalty**: Penalty cho tư thế tay không tự nhiên

## 🎯 Hệ thống hỗ trợ

### 1. Assist System
- **Mục đích**: Hỗ trợ robot đứng vững
- **Cường độ**: 0.8 (mạnh)
- **Decay**: 0.995 (giảm chậm)
- **Tác dụng**: Đưa joints về vị trí mặc định

### 2. Governor System
- **Mục đích**: Kiểm soát hành động
- **Max base velocity**: 2.0 m/s
- **Max pitch angle**: 0.3 rad
- **Tác dụng**: Giới hạn tốc độ và góc nghiêng

### 3. Termination Conditions
- **Fall height**: < 0.20m
- **Uprightness**: < 0.3
- **Timeout**: 1000 steps (20 giây)

## 📊 Kết quả mong đợi

### Success Criteria
- **Height**: ≥ 0.7m (gần chiều cao thực tế 0.793m)
- **Uprightness**: ≥ 0.9
- **Stability**: Không ngã trong 20 giây

### Performance Metrics
- **Success rate**: > 80%
- **Average standing time**: > 15 giây
- **Energy efficiency**: Thấp (penalty nhẹ)

## 🔍 Debugging & Troubleshooting

### 1. Model không load được
```bash
# Kiểm tra file tồn tại
ls -la model/
# Kiểm tra VecNormalize
ls -la model/*.pkl
```

### 2. MuJoCo viewer không hiển thị
```bash
# Cài đặt lại MuJoCo
pip install --upgrade mujoco mujoco-python-viewer
```

### 3. Training không hội tụ
- Kiểm tra reward function
- Điều chỉnh learning rate
- Tăng số timesteps

## 📝 Ghi chú quan trọng

⚠️ **Lưu ý**: Dự án này được thiết kế đặc biệt cho việc **standing** (đứng thẳng), không phải walking hay jumping. Robot được cấu hình để:

- ✅ Đứng ở chiều cao thực tế (0.793m)
- ✅ Giữ thăng bằng ổn định
- ✅ Không nhảy lên cao
- ✅ Tư thế tự nhiên (tay không dang rộng)

## 🤝 Đóng góp

Nếu bạn muốn cải thiện dự án:

1. **Fork** repository
2. **Tạo branch** mới cho feature
3. **Commit** changes
4. **Push** lên branch
5. **Tạo Pull Request**

## 📄 License

Dự án này sử dụng các thư viện open source:
- **MuJoCo**: Apache 2.0
- **Stable Baselines3**: MIT
- **Gymnasium**: MIT

## 🙏 Acknowledgments

- **Unitree Robotics** cho robot model G1
- **MuJoCo team** cho physics engine
- **Stable Baselines3 team** cho RL algorithms
- **OpenAI** cho Gym/Gymnasium framework

---

**Tác giả**: [Tên của bạn]  
**Ngày tạo**: [Ngày]  
**Phiên bản**: 1.0.0  
**Trạng thái**: ✅ Hoàn thành - Robot đã học được standing!