import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle
import time
import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random

# ==================== 常量定义 ====================
class DQNConfig:
    """DQN算法配置常量"""
    # 环境配置
    ENV_NAME = 'FrozenLake-v1'
    MAP_NAME = "8x8"  # 8x8地图
    IS_SLIPPERY = False  # 关闭滑冰，让智能体学会基本策略
    
    # 网络结构配置
    INPUT_SIZE = 64  # 8x8 = 64个状态
    HIDDEN_SIZE = 128  # 隐藏层大小
    OUTPUT_SIZE = 4   # 4个动作
    NUM_HIDDEN_LAYERS = 2  # 隐藏层数量
    
    # 学习参数
    LEARNING_RATE = 0.001
    GAMMA = 0.95  # 折扣因子
    EPSILON_START = 1.0
    EPSILON_END = 0.01
    EPSILON_DECAY = 0.995
    
    # 经验回放配置
    REPLAY_BUFFER_SIZE = 10000
    BATCH_SIZE = 32
    MIN_REPLAY_SIZE = 1000  # 开始训练前的最小经验数量
    
    # 目标网络配置
    TARGET_UPDATE_FREQUENCY = 100  # 每100步更新一次目标网络
    
    # 训练配置
    DEFAULT_TRAINING_EPISODES = 10000
    DEMO_EPISODES = 100
    MAX_STEPS_PER_EPISODE = 200
    
    # 奖励塑造
    HOLE_PENALTY = -0.1
    SUCCESS_REWARD = 1.0
    STEP_PENALTY = -0.001
    
    # 检查点配置
    CHECKPOINT_INTERVAL = 500
    MODEL_FILE = 'frozen_lake_dqn_model.pkl'
    CHECKPOINT_FILE = 'frozenlake_dqn_checkpoint.pkl'
    PLOT_FILE = 'frozen_lake_dqn_progress.png'
    
    # 设备配置
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 显示配置
    LOG_INTERVAL = 100
    SEPARATOR_LENGTH = 80
    DEMO_SEPARATOR_LENGTH = 50
    ACTION_NAMES = ["左", "下", "右", "上"]

# ==================== DQN网络模型 ====================
class DQNNetwork(nn.Module):
    """DQN神经网络模型"""
    
    def __init__(self, input_size, hidden_size, output_size, num_hidden_layers=2):
        super(DQNNetwork, self).__init__()
        
        # 输入层
        self.input_layer = nn.Linear(input_size, hidden_size)
        
        # 隐藏层
        self.hidden_layers = nn.ModuleList()
        for _ in range(num_hidden_layers - 1):
            self.hidden_layers.append(nn.Linear(hidden_size, hidden_size))
        
        # 输出层
        self.output_layer = nn.Linear(hidden_size, output_size)
        
        # Dropout层（可选，用于防止过拟合）
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        """前向传播"""
        # 输入层 + ReLU激活
        x = F.relu(self.input_layer(x))
        x = self.dropout(x)
        
        # 隐藏层 + ReLU激活
        for hidden_layer in self.hidden_layers:
            x = F.relu(hidden_layer(x))
            x = self.dropout(x)
        
        # 输出层（不激活，直接输出Q值）
        x = self.output_layer(x)
        return x

# ==================== 经验回放缓冲区 ====================
class ReplayBuffer:
    """经验回放缓冲区"""
    
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """添加经验到缓冲区"""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """从缓冲区随机采样一批经验"""
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)

# ==================== DQN智能体 ====================
class DQNAgent:
    """DQN智能体"""
    
    def __init__(self, config):
        self.config = config
        self.device = config.DEVICE
        
        # 创建主网络和目标网络
        self.q_network = DQNNetwork(
            config.INPUT_SIZE, 
            config.HIDDEN_SIZE, 
            config.OUTPUT_SIZE,
            config.NUM_HIDDEN_LAYERS
        ).to(self.device)
        
        self.target_network = DQNNetwork(
            config.INPUT_SIZE, 
            config.HIDDEN_SIZE, 
            config.OUTPUT_SIZE,
            config.NUM_HIDDEN_LAYERS
        ).to(self.device)
        
        # 复制主网络参数到目标网络
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # 优化器
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=config.LEARNING_RATE)
        
        # 经验回放缓冲区
        self.replay_buffer = ReplayBuffer(config.REPLAY_BUFFER_SIZE)
        
        # 探索率
        self.epsilon = config.EPSILON_START
        
        # 训练步数计数器
        self.step_count = 0
        
    def state_to_tensor(self, state):
        """将状态转换为one-hot张量"""
        state_tensor = torch.zeros(self.config.INPUT_SIZE, dtype=torch.float32)
        state_tensor[state] = 1.0
        return state_tensor.unsqueeze(0).to(self.device)
    
    def select_action(self, state, training=True):
        """选择动作（ε-贪婪策略）"""
        if training and random.random() < self.epsilon:
            # 随机探索
            return random.randint(0, self.config.OUTPUT_SIZE - 1)
        else:
            # 贪婪选择
            with torch.no_grad():
                state_tensor = self.state_to_tensor(state)
                q_values = self.q_network(state_tensor)
                return q_values.argmax().item()
    
    def store_experience(self, state, action, reward, next_state, done):
        """存储经验到回放缓冲区"""
        state_tensor = self.state_to_tensor(state).squeeze(0).cpu().numpy()
        next_state_tensor = self.state_to_tensor(next_state).squeeze(0).cpu().numpy()
        
        self.replay_buffer.push(state_tensor, action, reward, next_state_tensor, done)
    
    def train(self):
        """训练网络"""
        if len(self.replay_buffer) < self.config.MIN_REPLAY_SIZE:
            return
        
        # 从回放缓冲区采样
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.config.BATCH_SIZE)
        
        # 转换为张量
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)
        
        # 计算当前Q值
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # 计算目标Q值
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (self.config.GAMMA * next_q_values * ~dones)
        
        # 计算损失
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # 更新目标网络
        self.step_count += 1
        if self.step_count % self.config.TARGET_UPDATE_FREQUENCY == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        # 衰减探索率
        if self.epsilon > self.config.EPSILON_END:
            self.epsilon *= self.config.EPSILON_DECAY
    
    def save_model(self, filepath, episode=None):
        """保存模型"""
        checkpoint = {
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'step_count': self.step_count
        }
        if episode is not None:
            checkpoint['episode'] = episode
        torch.save(checkpoint, filepath)
    
    def load_model(self, filepath):
        """加载模型"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.step_count = checkpoint['step_count']

# ==================== 辅助函数 ====================
def create_environment(render=False):
    """创建FrozenLake环境"""
    try:
        env = gym.make(
            DQNConfig.ENV_NAME, 
            map_name=DQNConfig.MAP_NAME, 
            is_slippery=DQNConfig.IS_SLIPPERY, 
            render_mode='human' if render else None
        )
        return env
    except Exception as e:
        print(f"❌ 创建环境失败: {e}")
        print("请确保已安装gymnasium和FrozenLake环境")
        return None

def shape_reward(reward, terminated, truncated):
    """奖励塑造"""
    shaped_reward = reward
    
    if terminated and reward == 0.0:  # 掉入冰洞
        shaped_reward += DQNConfig.HOLE_PENALTY
    elif not terminated and not truncated:  # 正常步骤
        shaped_reward += DQNConfig.STEP_PENALTY
    
    return shaped_reward

# ==================== 主训练函数 ====================
def train_dqn(episodes, render=False, log_details=True, checkpoint_file=None):
    """
    训练DQN智能体
    
    Args:
        episodes: 训练回合数
        render: 是否显示可视化界面
        log_details: 是否显示详细日志
        checkpoint_file: 检查点文件路径
    """
    if checkpoint_file is None:
        checkpoint_file = DQNConfig.CHECKPOINT_FILE
    
    # 创建环境和智能体
    env = create_environment(render)
    if env is None:
        return
    
    agent = DQNAgent(DQNConfig)
    
    # 尝试加载检查点
    if os.path.exists(checkpoint_file):
        try:
            agent.load_model(checkpoint_file)
            print(f"✅ 加载检查点: 步数={agent.step_count}, ε={agent.epsilon:.4f}")
        except Exception as e:
            print(f"⚠️ 加载检查点失败: {e}, 从头开始训练")
    
    # 训练统计
    rewards_per_episode = []
    success_count = 0
    
    if log_details:
        print("=" * DQNConfig.SEPARATOR_LENGTH)
        print("🧠 DQN 算法训练开始")
        print(f"📊 环境: FrozenLake 8x8, 总回合数: {episodes}")
        print(f"🔧 设备: {DQNConfig.DEVICE}")
        print(f"🧠 网络结构: {DQNConfig.INPUT_SIZE}→{DQNConfig.HIDDEN_SIZE}→{DQNConfig.OUTPUT_SIZE}")
        print(f"📚 回放缓冲区: {DQNConfig.REPLAY_BUFFER_SIZE}, 批次大小: {DQNConfig.BATCH_SIZE}")
        print("=" * DQNConfig.SEPARATOR_LENGTH)
    else:
        print(f"🧠 开始DQN训练: {episodes}回合, 设备: {DQNConfig.DEVICE}")
        print("💡 训练过程中会显示进度信息...")
    
    for episode in range(episodes):
        state = env.reset()[0]
        episode_reward = 0
        episode_steps = 0
        
        # 显示训练进度（每100回合显示一次）
        if not log_details and (episode + 1) % 100 == 0:
            progress = (episode + 1) / episodes * 100
            print(f'\r🚀 训练进度: {progress:.1f}% ({episode + 1}/{episodes})', end='', flush=True)
        
        while episode_steps < DQNConfig.MAX_STEPS_PER_EPISODE:
            # 选择动作
            action = agent.select_action(state, training=True)
            
            # 执行动作
            next_state, reward, terminated, truncated, _ = env.step(action)
            
            # 奖励塑造
            shaped_reward = shape_reward(reward, terminated, truncated)
            
            # 存储经验
            agent.store_experience(state, action, shaped_reward, next_state, terminated or truncated)
            
            # 训练网络
            agent.train()
            
            # 更新状态和统计
            state = next_state
            episode_reward += reward
            episode_steps += 1
            
            if terminated or truncated:
                break
        
        # 记录成功
        if reward > 0:
            success_count += 1
        
        rewards_per_episode.append(episode_reward)
        
        # 显示进度
        if log_details and (episode + 1) % DQNConfig.LOG_INTERVAL == 0:
            avg_reward = np.mean(rewards_per_episode[-DQNConfig.LOG_INTERVAL:])
            success_rate = success_count / (episode + 1) * 100
            print(f"回合 {episode + 1}/{episodes}: "
                  f"平均奖励={avg_reward:.3f}, 成功率={success_rate:.1f}%, "
                  f"ε={agent.epsilon:.4f}, 步数={agent.step_count}")
        
        # 保存检查点
        if (episode + 1) % DQNConfig.CHECKPOINT_INTERVAL == 0:
            agent.save_model(checkpoint_file, episode + 1)
            if log_details:
                print(f"💾 已保存检查点到 {checkpoint_file}")
    
    env.close()
    
    # 训练完成后换行，让进度条显示完整
    if not log_details:
        print()  # 换行
    
    # 训练完成总结
    if log_details:
        final_success_rate = success_count / episodes * 100
        avg_final_reward = np.mean(rewards_per_episode[-100:]) if len(rewards_per_episode) >= 100 else np.mean(rewards_per_episode)
        
        print("\n" + "=" * DQNConfig.SEPARATOR_LENGTH)
        print("🏁 DQN训练完成!")
        print(f"📊 最终统计:")
        print(f"   • 总回合数: {episodes}")
        print(f"   • 成功率: {final_success_rate:.2f}%")
        print(f"   • 最后100回合平均奖励: {avg_final_reward:.3f}")
        print(f"   • 最终探索率: {agent.epsilon:.4f}")
        print(f"   • 总训练步数: {agent.step_count}")
        print("=" * DQNConfig.SEPARATOR_LENGTH)
    
    # 保存最终模型
    agent.save_model(DQNConfig.MODEL_FILE, episodes)
    if log_details:
        print(f"💾 最终模型已保存到 {DQNConfig.MODEL_FILE}")
    
    # 绘制训练曲线
    plot_training_progress(rewards_per_episode, log_details)
    
    return agent

def plot_training_progress(rewards_per_episode, log_details=True):
    """绘制训练进度图"""
    plt.figure(figsize=(12, 8))
    
    # 移动平均奖励
    window_size = 100
    if len(rewards_per_episode) >= window_size:
        moving_avg = np.convolve(rewards_per_episode, np.ones(window_size)/window_size, mode='valid')
        plt.subplot(2, 1, 1)
        plt.plot(moving_avg)
        plt.title(f'DQN训练过程 - 移动平均奖励 (窗口大小: {window_size})', fontsize=14)
        plt.xlabel('回合数')
        plt.ylabel('平均奖励')
        plt.grid(True, alpha=0.3)
    
    # 成功率曲线
    success_rate = []
    success_count = 0
    for i, reward in enumerate(rewards_per_episode):
        if reward > 0:
            success_count += 1
        success_rate.append(success_count / (i + 1) * 100)
    
    plt.subplot(2, 1, 2)
    plt.plot(success_rate, 'r-', linewidth=2)
    plt.title('累积成功率变化', fontsize=14)
    plt.xlabel('回合数')
    plt.ylabel('成功率 (%)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    try:
        plt.savefig(DQNConfig.PLOT_FILE, dpi=150, bbox_inches='tight')
        if log_details:
            print(f"📈 训练图表已保存到 {DQNConfig.PLOT_FILE}")
    except Exception as e:
        print(f"⚠️ 保存图表失败: {e}")
    
    plt.close()

def demo_trained_agent(episodes=100, render=True, log_details=True):
    """演示训练好的智能体"""
    # 创建环境和智能体
    env = create_environment(render)
    if env is None:
        return
    
    agent = DQNAgent(DQNConfig)
    
    # 加载训练好的模型
    try:
        agent.load_model(DQNConfig.MODEL_FILE)
        print(f"✅ 加载训练好的模型: 步数={agent.step_count}")
    except FileNotFoundError:
        print(f"❌ 未找到训练好的模型文件 '{DQNConfig.MODEL_FILE}'")
        print("请先运行训练模式")
        return
    
    # 设置测试模式（不探索）
    agent.epsilon = 0.0
    
    success_count = 0
    total_reward = 0
    
    if log_details:
        print("\n" + "=" * DQNConfig.DEMO_SEPARATOR_LENGTH)
        print("🎮 演示训练好的DQN智能体")
        print("=" * DQNConfig.DEMO_SEPARATOR_LENGTH)
    
    for episode in range(episodes):
        state = env.reset()[0]
        episode_reward = 0
        episode_steps = 0
        
        if log_details and episode < 5:  # 只显示前5个回合的详细信息
            print(f"\n📍 演示回合 {episode + 1}")
        
        while episode_steps < DQNConfig.MAX_STEPS_PER_EPISODE:
            # 选择动作（贪婪策略）
            action = agent.select_action(state, training=False)
            
            if log_details and episode < 5:
                print(f"  步骤 {episode_steps + 1}: 状态 {state} → 动作 '{DQNConfig.ACTION_NAMES[action]}'")
            
            # 执行动作
            next_state, reward, terminated, truncated, _ = env.step(action)
            
            state = next_state
            episode_reward += reward
            episode_steps += 1
            
            if terminated or truncated:
                break
        
        if reward > 0:
            success_count += 1
            if log_details and episode < 5:
                print(f"  🎉 成功到达目标! 奖励: {reward}")
        else:
            if log_details and episode < 5:
                print(f"  ❄️ 掉入冰洞或超时")
        
        total_reward += episode_reward
        
        if log_details and episode < 5:
            print(f"  回合总结: {episode_steps}步, 奖励={episode_reward}")
    
    env.close()
    
    # 演示总结
    success_rate = success_count / episodes * 100
    avg_reward = total_reward / episodes
    
    if log_details:
        print(f"\n📊 演示总结:")
        print(f"   • 演示回合数: {episodes}")
        print(f"   • 成功率: {success_rate:.1f}%")
        print(f"   • 平均奖励: {avg_reward:.3f}")
        print("=" * DQNConfig.DEMO_SEPARATOR_LENGTH)

if __name__ == '__main__':
    print("🚀 启动 DQN 算法演示")
    print("💡 提示: render=True 显示可视化界面，render=False 加快训练速度")
    print("💡 提示: log_details=False 可以关闭详细日志")
    print()
    
    # 检查是否有之前的检查点
    checkpoint_file = DQNConfig.CHECKPOINT_FILE
    if os.path.exists(checkpoint_file):
        try:
            checkpoint = torch.load(checkpoint_file, map_location=DQNConfig.DEVICE)
            print(f"🔍 发现之前的检查点: 已训练 {checkpoint['step_count']} 步")
            print(f"   探索率: {checkpoint['epsilon']:.4f}")
            if 'episode' in checkpoint:
                print(f"   已训练回合: {checkpoint['episode']}")
            
            # 询问用户选择
            print("\n请选择操作:")
            print("1. 直接演示训练好的模型")
            print("2. 继续训练更多回合")
            print("3. 重新开始训练")
            
            choice = input("请输入选择 (1/2/3): ").strip()
            
            if choice == '1':
                print("✅ 直接进入演示模式...")
                # 跳过训练，直接进入演示
                pass
            elif choice == '2':
                print("✅ 继续训练...")
                try:
                    episodes = int(input(f"请输入要训练的回合数 (默认 {DQNConfig.DEFAULT_TRAINING_EPISODES}): ") or DQNConfig.DEFAULT_TRAINING_EPISODES)
                    if episodes > 0:
                        print(f"📊 开始训练 {episodes} 回合...")
                        train_dqn(episodes=episodes, 
                                 render=False, log_details=False, checkpoint_file=checkpoint_file)
                    else:
                        print("❌ 回合数必须大于0，进入演示模式")
                except ValueError:
                    print("❌ 请输入有效的数字，进入演示模式")
            elif choice == '3':
                print("🆕 开始新训练...")
                train_dqn(episodes=DQNConfig.DEFAULT_TRAINING_EPISODES, 
                         render=False, log_details=False)
            else:
                print("❌ 无效选择，默认进入演示模式")
                # 默认进入演示模式
        except Exception as e:
            print(f"⚠️ 加载检查点失败: {e}, 重新开始训练")
            train_dqn(episodes=DQNConfig.DEFAULT_TRAINING_EPISODES, 
                     render=False, log_details=False)
    else:
        print("🆕 开始新训练...")
        train_dqn(episodes=DQNConfig.DEFAULT_TRAINING_EPISODES, 
                 render=False, log_details=False)
    
    print("\n✅ 训练完成！现在开始可视化演示...")
    input("按Enter键开始演示...")
    
    # 演示训练好的智能体
    try:
        demo_trained_agent(episodes=DQNConfig.DEMO_EPISODES, render=True, log_details=True)
    except KeyboardInterrupt:
        print("\n👋 演示结束")
