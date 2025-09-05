"""
Script so sánh environment cũ vs mới
Test performance của cả hai implementation
"""

import numpy as np
import matplotlib.pyplot as plt
from g1_standing_env_improved import G1StandingEnvImproved
from g1_standing_env_fixed import G1StandingEnvFixed


def test_environment(env_class, env_name, num_episodes=10):
    """Test một environment và trả về metrics"""
    print(f"\n🧪 Testing {env_name}...")
    
    try:
        if env_class == G1StandingEnvImproved:
            env = env_class(
                xml_path="g1_description/g1_23dof.xml",
                config_path="configs/g1_train_stand_v3.yaml",
                max_episode_steps=1000
            )
        else:
            env = env_class(
                xml_path="g1_description/g1_23dof.xml",
                config_path="configs/g1_train_stand_fixed.yaml",
                max_episode_steps=1000
            )
        
        episode_rewards = []
        episode_heights = []
        episode_durations = []
        episode_successes = []
        
        for episode in range(num_episodes):
            obs, info = env.reset()
            total_reward = 0
            max_height = 0
            steps = 0
            
            for step in range(1000):
                # Random action
                action = env.action_space.sample() * 0.1
                obs, reward, done, truncated, info = env.step(action)
                
                total_reward += reward
                height = info.get('height', 0)
                max_height = max(max_height, height)
                steps += 1
                
                if done or truncated:
                    break
            
            episode_rewards.append(total_reward)
            episode_heights.append(max_height)
            episode_durations.append(steps)
            episode_successes.append(1 if max_height > 0.4 else 0)
            
            if episode % 2 == 0:
                print(f"  Episode {episode}: reward={total_reward:.2f}, "
                      f"max_height={max_height:.3f}, steps={steps}")
        
        env.close()
        
        # Tính metrics
        avg_reward = np.mean(episode_rewards)
        avg_height = np.mean(episode_heights)
        avg_duration = np.mean(episode_durations)
        success_rate = np.mean(episode_successes)
        
        print(f"✅ {env_name} Results:")
        print(f"   - Avg reward: {avg_reward:.3f}")
        print(f"   - Avg max height: {avg_height:.3f}")
        print(f"   - Avg duration: {avg_duration:.1f} steps")
        print(f"   - Success rate: {success_rate:.1%}")
        
        return {
            'rewards': episode_rewards,
            'heights': episode_heights,
            'durations': episode_durations,
            'successes': episode_successes,
            'avg_reward': avg_reward,
            'avg_height': avg_height,
            'avg_duration': avg_duration,
            'success_rate': success_rate
        }
        
    except Exception as e:
        print(f"❌ Error testing {env_name}: {e}")
        return None


def plot_comparison(results_old, results_new):
    """Vẽ biểu đồ so sánh"""
    if results_old is None or results_new is None:
        print("❌ Không thể vẽ biểu đồ do lỗi test")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Environment Comparison: Old vs Fixed', fontsize=16)
    
    # Reward comparison
    axes[0, 0].hist([results_old['rewards'], results_new['rewards']], 
                    bins=10, alpha=0.7, label=['Old', 'Fixed'])
    axes[0, 0].set_title('Reward Distribution')
    axes[0, 0].set_xlabel('Total Reward')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].legend()
    
    # Height comparison
    axes[0, 1].hist([results_old['heights'], results_new['heights']], 
                    bins=10, alpha=0.7, label=['Old', 'Fixed'])
    axes[0, 1].set_title('Max Height Distribution')
    axes[0, 1].set_xlabel('Max Height (m)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].legend()
    
    # Duration comparison
    axes[1, 0].hist([results_old['durations'], results_new['durations']], 
                    bins=10, alpha=0.7, label=['Old', 'Fixed'])
    axes[1, 0].set_title('Episode Duration Distribution')
    axes[1, 0].set_xlabel('Steps')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].legend()
    
    # Success rate comparison
    success_rates = [results_old['success_rate'], results_new['success_rate']]
    axes[1, 1].bar(['Old', 'Fixed'], success_rates, alpha=0.7, color=['red', 'green'])
    axes[1, 1].set_title('Success Rate Comparison')
    axes[1, 1].set_ylabel('Success Rate')
    axes[1, 1].set_ylim(0, 1)
    
    # Add value labels on bars
    for i, v in enumerate(success_rates):
        axes[1, 1].text(i, v + 0.01, f'{v:.1%}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('environment_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n📊 Biểu đồ đã lưu: environment_comparison.png")


def main():
    """Main function"""
    print("🔬 SO SÁNH ENVIRONMENT: OLD vs FIXED")
    print("=" * 50)
    
    # Test old environment
    results_old = test_environment(
        G1StandingEnvImproved, 
        "OLD (Improved)", 
        num_episodes=5
    )
    
    # Test new environment
    results_new = test_environment(
        G1StandingEnvFixed, 
        "FIXED (PD + Assist)", 
        num_episodes=5
    )
    
    # So sánh kết quả
    if results_old and results_new:
        print(f"\n📊 SO SÁNH KẾT QUẢ:")
        print(f"{'Metric':<20} {'Old':<10} {'Fixed':<10} {'Improvement':<15}")
        print("-" * 60)
        
        reward_improvement = ((results_new['avg_reward'] - results_old['avg_reward']) / 
                            abs(results_old['avg_reward']) * 100) if results_old['avg_reward'] != 0 else 0
        height_improvement = ((results_new['avg_height'] - results_old['avg_height']) / 
                            abs(results_old['avg_height']) * 100) if results_old['avg_height'] != 0 else 0
        success_improvement = ((results_new['success_rate'] - results_old['success_rate']) / 
                              abs(results_old['success_rate']) * 100) if results_old['success_rate'] != 0 else 0
        
        print(f"{'Avg Reward':<20} {results_old['avg_reward']:<10.3f} {results_new['avg_reward']:<10.3f} {reward_improvement:+.1f}%")
        print(f"{'Avg Height':<20} {results_old['avg_height']:<10.3f} {results_new['avg_height']:<10.3f} {height_improvement:+.1f}%")
        print(f"{'Success Rate':<20} {results_old['success_rate']:<10.1%} {results_new['success_rate']:<10.1%} {success_improvement:+.1f}%")
        print(f"{'Avg Duration':<20} {results_old['avg_duration']:<10.1f} {results_new['avg_duration']:<10.1f} {'N/A':<15}")
        
        # Vẽ biểu đồ
        plot_comparison(results_old, results_new)
        
        # Kết luận
        print(f"\n🎯 KẾT LUẬN:")
        if results_new['success_rate'] > results_old['success_rate']:
            print(f"✅ Environment FIXED hoạt động TỐT HƠN!")
            print(f"   - Success rate tăng {success_improvement:+.1f}%")
            print(f"   - Avg height tăng {height_improvement:+.1f}%")
        else:
            print(f"⚠️  Environment FIXED cần điều chỉnh thêm")
            print(f"   - Success rate: {success_improvement:+.1f}%")
            print(f"   - Avg height: {height_improvement:+.1f}%")
    
    else:
        print(f"❌ Không thể so sánh do lỗi test")


if __name__ == "__main__":
    main()
