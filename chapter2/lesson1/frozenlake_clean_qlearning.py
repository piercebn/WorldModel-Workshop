"""
朴素Q-Learning算法实现 - 专注于展示原理
使用最佳实践的代码结构，简洁明了地展示Q-Learning的核心思想
"""

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle
import time
import random
from typing import Tuple, Optional
import os

class QLearningAgent:
    """朴素的Q-Learning智能体"""
    
    def __init__(self, 
                 state_size: int, 
                 action_size: int,
                 learning_rate: float = 0.1,
                 discount_factor: float = 0.95,
                 epsilon: float = 1.0,
                 epsilon_decay: float = 0.995,
                 epsilon_min: float = 0.01,
                 seed: int = 42):
        """
        初始化Q-Learning智能体
        
        Args:
            state_size: 状态空间大小
            action_size: 动作空间大小
            learning_rate: 学习率 (alpha)
            discount_factor: 折扣因子 (gamma)
            epsilon: 初始探索率
            epsilon_decay: 探索率衰减率
            epsilon_min: 最小探索率
            seed: 随机种子
        """
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # 初始化Q表
        np.random.seed(seed)
        self.q_table = np.zeros((state_size, action_size))
        
        # 统计信息
        self.episode_rewards = []
        self.episode_lengths = []
        self.success_rates = []
        
    def select_action(self, state: int, training: bool = True) -> int:
        """
        使用epsilon-greedy策略选择动作
        
        Args:
            state: 当前状态
            training: 是否在训练模式
            
        Returns:
            选择的动作
        """
        if training and np.random.random() < self.epsilon:
            # 探索：随机选择动作
            return np.random.choice(self.action_size)
        else:
            # 利用：选择Q值最大的动作
            return np.argmax(self.q_table[state])
    
    def update_q_table(self, state: int, action: int, reward: float, 
                      next_state: int, done: bool):
        """
        更新Q表
        
        Args:
            state: 当前状态
            action: 执行的动作
            reward: 获得的奖励
            next_state: 下一个状态
            done: 是否结束
        """
        # Q-Learning更新公式
        current_q = self.q_table[state, action]
        
        if done:
            # 如果回合结束，目标值就是奖励
            target = reward
        else:
            # 否则使用贝尔曼方程
            max_next_q = np.max(self.q_table[next_state])
            target = reward + self.discount_factor * max_next_q
        
        # 更新Q值
        self.q_table[state, action] = current_q + self.learning_rate * (target - current_q)
    
    def decay_epsilon(self):
        """衰减探索率"""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def save_model(self, filepath: str):
        """保存模型"""
        model_data = {
            'q_table': self.q_table,
            'epsilon': self.epsilon,
            'learning_rate': self.learning_rate,
            'discount_factor': self.discount_factor,
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'success_rates': self.success_rates
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_model(self, filepath: str):
        """加载模型"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.q_table = model_data['q_table']
        self.epsilon = model_data['epsilon']
        self.learning_rate = model_data['learning_rate']
        self.discount_factor = model_data['discount_factor']
        self.episode_rewards = model_data.get('episode_rewards', [])
        self.episode_lengths = model_data.get('episode_lengths', [])
        self.success_rates = model_data.get('success_rates', [])

def train_qlearning(env_id: str = "FrozenLake-v1",
                   total_episodes: int = 5000,
                   learning_rate: float = 0.1,
                   discount_factor: float = 0.95,
                   epsilon: float = 1.0,
                   epsilon_decay: float = 0.995,
                   epsilon_min: float = 0.01,
                   eval_freq: int = 500,
                   seed: int = 42,
                   verbose: bool = True):
    """
    训练Q-Learning智能体
    
    Args:
        env_id: 环境ID
        total_episodes: 总训练回合数
        learning_rate: 学习率
        discount_factor: 折扣因子
        epsilon: 初始探索率
        epsilon_decay: 探索率衰减率
        epsilon_min: 最小探索率
        eval_freq: 评估频率
        save_freq: 保存频率
        seed: 随机种子
        verbose: 是否显示详细信息
    """
    
    # 设置随机种子
    random.seed(seed)
    np.random.seed(seed)
    
    # 创建环境
    env = gym.make(env_id, map_name="8x8", is_slippery=True)
    
    # 创建智能体
    agent = QLearningAgent(
        state_size=env.observation_space.n,
        action_size=env.action_space.n,
        learning_rate=learning_rate,
        discount_factor=discount_factor,
        epsilon=epsilon,
        epsilon_decay=epsilon_decay,
        epsilon_min=epsilon_min,
        seed=seed
    )
    
    # 检查是否有预训练模型
    model_path = "frozenlake_qlearning_model.pkl"
    if os.path.exists(model_path):
        if verbose:
            print(f"🔍 发现预训练模型: {model_path}")
        try:
            agent.load_model(model_path)
            if verbose:
                print("✅ 成功加载预训练模型")
        except Exception as e:
            if verbose:
                print(f"⚠️  加载模型失败: {e}，重新开始训练")
    
    if verbose:
        print("🚀 开始Q-Learning训练...")
        print(f"📊 训练参数:")
        print(f"   • 环境: {env_id} (8x8)")
        print(f"   • 总回合数: {total_episodes:,}")
        print(f"   • 学习率: {learning_rate}")
        print(f"   • 折扣因子: {discount_factor}")
        print(f"   • 初始探索率: {epsilon}")
        print(f"   • 探索率衰减: {epsilon_decay}")
        print(f"   • 最小探索率: {epsilon_min}")
        print("=" * 60)
    
    start_time = time.time()
    
    for episode in range(total_episodes):
        state, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        
        while not done:
            # 选择动作
            action = agent.select_action(state, training=True)
            
            # 执行动作
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # 更新Q表
            agent.update_q_table(state, action, reward, next_state, done)
            
            # 更新状态和统计
            state = next_state
            episode_reward += reward
            episode_length += 1
        
        # 记录统计信息
        agent.episode_rewards.append(episode_reward)
        agent.episode_lengths.append(episode_length)
        
        # 计算成功率
        recent_rewards = agent.episode_rewards[-100:]
        success_rate = sum(1 for r in recent_rewards if r > 0) / len(recent_rewards) * 100
        agent.success_rates.append(success_rate)
        
        # 衰减探索率
        agent.decay_epsilon()
        
        # 定期输出进度
        if verbose and (episode + 1) % eval_freq == 0:
            avg_reward = np.mean(recent_rewards)
            avg_length = np.mean(agent.episode_lengths[-100:])
            print(f"📊 回合 {episode + 1:,}/{total_episodes:,}: "
                  f"成功率={success_rate:.1f}%, "
                  f"平均奖励={avg_reward:.3f}, "
                  f"平均步数={avg_length:.1f}, "
                  f"ε={agent.epsilon:.4f}")
        
        # 定期保存模型
        if (episode + 1) % 1000 == 0:
            agent.save_model(f"frozenlake_qlearning_checkpoint_{episode + 1}.pkl")
            if verbose:
                print(f"💾 已保存checkpoint: frozenlake_qlearning_checkpoint_{episode + 1}.pkl")
    
    training_time = time.time() - start_time
    
    # 保存最终模型
    agent.save_model(model_path)
    
    if verbose:
        final_success_rate = agent.success_rates[-1] if agent.success_rates else 0
        print(f"\n🏁 训练完成!")
        print(f"⏰ 训练用时: {training_time:.2f}秒")
        print(f"📈 最终成功率: {final_success_rate:.1f}%")
        print(f"💾 模型已保存到: {model_path}")
    
    env.close()
    return agent

def evaluate_agent(agent: QLearningAgent, 
                  env_id: str = "FrozenLake-v1",
                  n_episodes: int = 100,
                  render: bool = False,
                  verbose: bool = True):
    """
    评估训练好的智能体
    
    Args:
        agent: 训练好的智能体
        env_id: 环境ID
        n_episodes: 评估回合数
        render: 是否渲染
        verbose: 是否显示详细信息
    """
    
    env = gym.make(env_id, map_name="8x8", is_slippery=True, 
                   render_mode='human' if render else None)
    
    episode_rewards = []
    episode_lengths = []
    success_count = 0
    
    if verbose:
        print(f"\n🎮 开始评估智能体 ({n_episodes}回合)...")
    
    for episode in range(n_episodes):
        state, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        
        while not done:
            # 使用确定性策略（不探索）
            action = agent.select_action(state, training=False)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            episode_length += 1
            
            if render:
                time.sleep(0.1)  # 减慢速度以便观察
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        if episode_reward > 0:
            success_count += 1
        
        if verbose and (episode + 1) % 20 == 0:
            current_success_rate = success_count / (episode + 1) * 100
            print(f"📊 进度 {episode + 1}/{n_episodes}: 成功率 {current_success_rate:.1f}%")
    
    env.close()
    
    # 计算统计数据
    success_rate = success_count / n_episodes * 100
    avg_reward = np.mean(episode_rewards)
    avg_length = np.mean(episode_lengths)
    
    if verbose:
        print(f"\n📈 评估结果:")
        print(f"   • 成功率: {success_rate:.1f}% ({success_count}/{n_episodes})")
        print(f"   • 平均奖励: {avg_reward:.3f}")
        print(f"   • 平均步数: {avg_length:.1f}")
    
    return {
        'success_rate': success_rate,
        'avg_reward': avg_reward,
        'avg_length': avg_length,
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths
    }

def plot_training_progress(agent: QLearningAgent, save_path: str = "frozenlake_qlearning_progress.png"):
    """绘制训练进度图表"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('CleanRL风格 Q-Learning 训练进度', fontsize=16, fontweight='bold')
    
    # 成功率曲线
    if agent.success_rates:
        axes[0, 0].plot(agent.success_rates, 'b-', linewidth=2)
        axes[0, 0].set_title('成功率变化 (100回合移动平均)')
        axes[0, 0].set_xlabel('回合数')
        axes[0, 0].set_ylabel('成功率 (%)')
        axes[0, 0].grid(True, alpha=0.3)
    
    # 回合奖励分布
    if agent.episode_rewards:
        axes[0, 1].hist(agent.episode_rewards, bins=3, alpha=0.7, color='green')
        axes[0, 1].set_title('回合奖励分布')
        axes[0, 1].set_xlabel('奖励')
        axes[0, 1].set_ylabel('频次')
        axes[0, 1].grid(True, alpha=0.3)
    
    # 回合长度变化
    if agent.episode_lengths:
        # 计算移动平均
        window_size = min(100, len(agent.episode_lengths) // 10)
        if window_size > 1:
            moving_avg = np.convolve(agent.episode_lengths, 
                                   np.ones(window_size)/window_size, mode='valid')
            axes[1, 0].plot(moving_avg, 'r-', linewidth=2, label=f'{window_size}回合移动平均')
            axes[1, 0].legend()
        else:
            axes[1, 0].plot(agent.episode_lengths, 'r-', alpha=0.6)
        
        axes[1, 0].set_title('回合步数变化')
        axes[1, 0].set_xlabel('回合数')
        axes[1, 0].set_ylabel('步数')
        axes[1, 0].grid(True, alpha=0.3)
    
    # 训练总结信息
    axes[1, 1].axis('off')
    final_success_rate = agent.success_rates[-1] if agent.success_rates else 0
    summary_text = f"""
    训练总结:
    
    • 总回合数: {len(agent.episode_rewards):,}
    • 最终成功率: {final_success_rate:.1f}%
    • 平均回合长度: {np.mean(agent.episode_lengths):.1f}步
    • 最终探索率: {agent.epsilon:.4f}
    • 学习率: {agent.learning_rate}
    • 折扣因子: {agent.discount_factor}
    
    算法: 朴素Q-Learning
    环境: FrozenLake-v1 (8x8)
    实现风格: CleanRL
    """
    
    axes[1, 1].text(0.1, 0.9, summary_text, transform=axes[1, 1].transAxes,
                   fontsize=12, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    plt.tight_layout()
    
    # 保存图表
    try:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"📈 训练进度图表已保存到: {save_path}")
    except Exception as e:
        print(f"⚠️  保存图表失败: {e}")
    
    plt.show()

def main():
    """主函数 - 简洁的Q-Learning演示"""
    
    print("=" * 60)
    print("🎯 朴素Q-Learning算法演示 - FrozenLake 8x8")
    print("专注于展示Q-Learning的核心原理和实现")
    print("=" * 60)
    
    # 训练智能体
    agent = train_qlearning(
        total_episodes=5000,
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01,
        eval_freq=500,
        seed=42,
        verbose=True
    )
    
    # 绘制训练进度
    plot_training_progress(agent)
    
    # 评估智能体
    print("\n" + "="*50)
    eval_results = evaluate_agent(
        agent, 
        n_episodes=100, 
        render=False,
        verbose=True
    )
    
    # 询问是否进行可视化演示
    try:
        response = input("\n🎮 是否进行可视化演示? (y/n): ").lower().strip()
        if response in ['y', 'yes', '是']:
            print("🎮 开始可视化演示...")
            evaluate_agent(agent, n_episodes=5, render=True, verbose=False)
    except KeyboardInterrupt:
        print("\n👋 程序结束")
    except:
        print("🎮 跳过可视化演示（非交互环境）")
    
    print("\n✅ 程序执行完成!")

if __name__ == "__main__":
    main()
