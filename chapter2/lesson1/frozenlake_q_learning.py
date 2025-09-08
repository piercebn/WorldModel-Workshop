import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle
import time
import sys
import os

# ==================== 常量定义 ====================
class QLearningConfig:
    """Q-Learning算法配置常量"""
    # 环境配置
    ENV_NAME = 'FrozenLake-v1'
    MAP_NAME = "8x8"  # 改为8x8地图
    IS_SLIPPERY = False  # 关闭滑冰，让智能体学会基本策略
    
    # 学习参数
    DEFAULT_LEARNING_RATE = 0.9
    DEFAULT_EPSILON = 1.0
    DEFAULT_DISCOUNT_FACTOR = 0.9
    DEFAULT_EPSILON_DECAY_RATE = 0.00001  # 进一步减慢探索率衰减
    
    # 奖励塑造
    HOLE_PENALTY = -0.1
    SUCCESS_REWARD = 1.0
    PROGRESS_REWARD = 0.01  # 向目标靠近的奖励
    STEP_PENALTY = -0.001   # 每步的小惩罚，鼓励尽快到达目标
    
    # 训练配置
    DEFAULT_TRAINING_EPISODES = 50000  # 8x8环境需要更多训练回合
    DEMO_EPISODES = 100
    
    # 检查点配置
    CHECKPOINT_INTERVAL_MAJOR = 20  # 每5%保存一次
    CHECKPOINT_INTERVAL_AUTO = 1000  # 每1000回合自动保存
    CHECKPOINT_INTERVAL_QUICK = 100  # 每100回合快速保存
    
    # 日志配置
    LOG_INTERVAL_PERCENT = 100  # 每1%输出一次日志
    DETAIL_LOG_STEPS = 15  # 详细日志显示步数
    DETAIL_LOG_EARLY_EPISODES = 5  # 前几回合显示详细日志
    
    # 进度条配置
    PROGRESS_BAR_LENGTH = 30
    
    # 学习率调整
    EPSILON_THRESHOLD = 0.01
    LEARNING_RATE_DECAY = 0.9999
    MIN_LEARNING_RATE = 0.01  # 保持1%的最小学习率，不要过度衰减
    MIN_EPSILON = 0.2  # 保持20%的最小探索率，增加探索
    
    # 图表配置
    PLOT_FIGURE_SIZE = (12, 8)
    PLOT_DPI = 150
    MOVING_AVERAGE_WINDOW = 100
    
    # 性能优化配置
    ENABLE_PERFORMANCE_OPTIMIZATION = True
    CACHE_Q_MAX = True  # 缓存Q值最大值计算
    
    # 文件配置
    MODEL_FILE = 'frozen_lake8x8.pkl'
    CHECKPOINT_FILE = 'frozenlake_checkpoint.pkl'
    PLOT_FILE = 'frozen_lake8x8.png'
    
    # 环境状态配置
    MAX_ACTIONS_PER_EPISODE = 200
    MAP_SIZE = 8  # 8x8地图
    
    # 动作名称
    ACTION_NAMES = ["左", "下", "右", "上"]
    
    # 验证配置
    MIN_EPSILON = 0.0
    MAX_EPSILON = 1.0
    MIN_LEARNING_RATE_VALID = 0.0
    MAX_LEARNING_RATE_VALID = 1.0
    
    # 显示配置
    STRATEGY_DISPLAY_ROWS = 4
    STRATEGY_DISPLAY_COLS = 4
    SEPARATOR_LENGTH = 80
    DEMO_SEPARATOR_LENGTH = 50
    LOG_SEPARATOR_LENGTH = 70

# ==================== 辅助函数 ====================
def validate_parameters(episodes):
    """验证输入参数"""
    if episodes <= 0:
        print("❌ 回合数必须大于0")
        return False
    
    if not isinstance(episodes, int):
        print("❌ 回合数必须是整数")
        return False
    
    return True

def create_environment(render=False):
    """创建FrozenLake环境"""
    try:
        env = gym.make(
            QLearningConfig.ENV_NAME, 
            map_name=QLearningConfig.MAP_NAME, 
            is_slippery=QLearningConfig.IS_SLIPPERY, 
            render_mode='human' if render else None
        )
        return env
    except Exception as e:
        print(f"❌ 创建环境失败: {e}")
        print("请确保已安装gymnasium和FrozenLake环境")
        return None

def load_checkpoint(checkpoint_file, env):
    """加载检查点文件"""
    try:
        if not os.path.isfile(checkpoint_file):
            print(f"⚠️  checkpoint路径不是文件: {checkpoint_file}")
            raise FileNotFoundError("Invalid checkpoint path")
            
        with open(checkpoint_file, 'rb') as f:
            checkpoint = pickle.load(f)
            
        # 验证checkpoint数据结构
        required_keys = ['q_table', 'episode', 'epsilon', 'learning_rate']
        if not all(key in checkpoint for key in required_keys):
            print(f"⚠️  checkpoint文件格式无效，缺少必要字段")
            raise ValueError("Invalid checkpoint format")
            
        q = checkpoint['q_table']
        start_episode = checkpoint['episode']
        epsilon = checkpoint['epsilon']
        learning_rate_a = checkpoint['learning_rate']
        
        # 验证Q表维度是否与环境匹配
        expected_shape = (env.observation_space.n, env.action_space.n)
        if q.shape != expected_shape:
            print(f"⚠️  Q表维度不匹配: 期望{expected_shape}, 实际{q.shape}")
            raise ValueError("Q table dimension mismatch")
        
        # 验证参数值的合理性
        if not (QLearningConfig.MIN_EPSILON <= epsilon <= QLearningConfig.MAX_EPSILON):
            print(f"⚠️  epsilon值不合理: {epsilon}, 重置为{QLearningConfig.DEFAULT_EPSILON}")
            epsilon = QLearningConfig.DEFAULT_EPSILON
        if not (QLearningConfig.MIN_LEARNING_RATE_VALID < learning_rate_a <= QLearningConfig.MAX_LEARNING_RATE_VALID):
            print(f"⚠️  学习率值不合理: {learning_rate_a}, 重置为{QLearningConfig.DEFAULT_LEARNING_RATE}")
            learning_rate_a = QLearningConfig.DEFAULT_LEARNING_RATE
        if start_episode < 0:
            print(f"⚠️  回合数不合理: {start_episode}, 重置为0")
            start_episode = 0
        
        print(f"✅ 加载checkpoint: 从第{start_episode}回合继续训练")
        print(f"   当前Q表状态: 平均Q值={np.mean(np.max(q, axis=1)):.4f}")
        return q, start_episode, epsilon, learning_rate_a
        
    except Exception as e:
        print(f"⚠️  加载checkpoint失败: {e}, 从头开始训练")
        return None, 0, QLearningConfig.DEFAULT_EPSILON, QLearningConfig.DEFAULT_LEARNING_RATE

def save_checkpoint(q, episode, epsilon, learning_rate, success_rate, avg_q, checkpoint_file):
    """保存检查点"""
    try:
        checkpoint = {
            'q_table': q,
            'episode': episode,
            'epsilon': epsilon,
            'learning_rate': learning_rate,
            'success_rate': success_rate,
            'avg_q': avg_q
        }
        with open(checkpoint_file, 'wb') as f:
            pickle.dump(checkpoint, f)
        return True
    except Exception as e:
        print(f"⚠️  保存checkpoint失败: {e}")
        return False

def run(episodes, is_training=True, render=False, log_details=True, checkpoint_file=None):
    """
    运行Q-Learning算法
    
    Args:
        episodes: 训练回合数
        is_training: 是否为训练模式
        render: 是否显示可视化界面
        log_details: 是否显示详细日志
        checkpoint_file: 检查点文件路径，默认为配置中的默认值
    """
    # 使用默认检查点文件
    if checkpoint_file is None:
        checkpoint_file = QLearningConfig.CHECKPOINT_FILE
    
    # 参数验证
    if not validate_parameters(episodes):
        return

    # 创建环境
    env = create_environment(render)
    if env is None:
        return

    # 初始化Q表和参数
    if is_training:
        if os.path.exists(checkpoint_file):
            q, start_episode, epsilon, learning_rate_a = load_checkpoint(checkpoint_file, env)
            if q is None:  # 加载失败，重新初始化
                # 使用小的随机值初始化Q表，而不是全零
                q = np.random.uniform(-0.1, 0.1, (env.observation_space.n, env.action_space.n))
        else:
            # 使用小的随机值初始化Q表，而不是全零
            q = np.random.uniform(-0.1, 0.1, (env.observation_space.n, env.action_space.n))
            start_episode = 0
            epsilon = QLearningConfig.DEFAULT_EPSILON
            learning_rate_a = QLearningConfig.DEFAULT_LEARNING_RATE
    else:
        # 测试模式，加载训练好的模型
        try:
            with open(QLearningConfig.MODEL_FILE, 'rb') as f:
                q = pickle.load(f)
            # 测试模式下也需要初始化epsilon
            epsilon = 0.0
            learning_rate_a = QLearningConfig.DEFAULT_LEARNING_RATE
            start_episode = 0
        except FileNotFoundError:
            print(f"❌ 未找到训练好的模型文件 '{QLearningConfig.MODEL_FILE}'")
            print("请先运行训练模式")
            return

    discount_factor_g = QLearningConfig.DEFAULT_DISCOUNT_FACTOR  # gamma or discount rate
    epsilon_decay_rate = QLearningConfig.DEFAULT_EPSILON_DECAY_RATE  # epsilon decay rate
    rng = np.random.default_rng()   # random number generator

    rewards_per_episode = np.zeros(episodes)
    
    # 日志相关变量
    action_names = QLearningConfig.ACTION_NAMES
    log_interval = max(1, episodes // QLearningConfig.LOG_INTERVAL_PERCENT)  # 每1%的进度输出一次详细日志
    step_count = 0
    
    if log_details and is_training:
        print("=" * QLearningConfig.SEPARATOR_LENGTH)
        print("🎯 Q-Learning 算法训练开始")
        print(f"📊 环境: FrozenLake 8x8, 总回合数: {episodes}")
        print(f"🧠 初始参数: α(学习率)={learning_rate_a}, γ(折扣因子)={discount_factor_g}, ε(探索率)={epsilon}")
        print("=" * QLearningConfig.SEPARATOR_LENGTH)

    for i in range(episodes):
        # 显示进度条
        if not log_details and is_training:
            # 计算总体进度（包括之前训练的回合数）
            total_episodes_trained = start_episode + i + 1
            total_target_episodes = start_episode + episodes  # 使用实际目标回合数
            progress = total_episodes_trained / total_target_episodes * 100
            bar_length = QLearningConfig.PROGRESS_BAR_LENGTH
            filled_length = int(bar_length * progress / 100)
            bar = '█' * filled_length + '░' * (bar_length - filled_length)
            print(f'\r🚀 训练进度: [{bar}] {progress:.1f}% ({total_episodes_trained}/{total_target_episodes})', end='', flush=True)
        
        state = env.reset()[0]  # states: 0 to 63, 0=top left corner,63=bottom right corner
        terminated = False      # True when fall in hole or reached goal
        truncated = False       # True when actions > MAX_ACTIONS_PER_EPISODE
        episode_steps = 0
        episode_reward = 0

        # 记录回合开始信息
        show_episode_detail = log_details and is_training and (i % log_interval == 0 or i < QLearningConfig.DETAIL_LOG_EARLY_EPISODES)
        
        if show_episode_detail:
            print(f"\n📍 回合 {i+1}/{episodes} (进度: {(i+1)/episodes*100:.1f}%)")
            print(f"🎲 当前探索率 ε = {epsilon:.4f}, 学习率 α = {learning_rate_a:.4f}")
            print(f"🗺️  起始位置: 状态 {state} (位置: 行{state//QLearningConfig.MAP_SIZE}, 列{state%QLearningConfig.MAP_SIZE})")

        while(not terminated and not truncated):
            step_count += 1
            episode_steps += 1
            
            # 详细日志 - 步骤开始
            if show_episode_detail and episode_steps <= QLearningConfig.DETAIL_LOG_STEPS:
                print(f"\n  📍 步骤 {episode_steps}: 当前状态 {state} (行{state//QLearningConfig.MAP_SIZE}, 列{state%QLearningConfig.MAP_SIZE})")
                print(f"     当前状态Q值: {q[state,:]}")
                sys.stdout.flush()
            
            # 动作选择策略
            random_value = rng.random() if is_training else 0
            if is_training and random_value < epsilon:
                action = env.action_space.sample() # actions: 0=left,1=down,2=right,3=up
                action_type = "🎲 随机探索"
                if show_episode_detail and episode_steps <= QLearningConfig.DETAIL_LOG_STEPS:
                    print(f"     决策依据: 随机数 {random_value:.4f} < ε({epsilon:.4f}) → 随机选择")
                    sys.stdout.flush()
            else:
                action = np.argmax(q[state,:])
                action_type = "🧠 策略选择"
                if show_episode_detail and episode_steps <= QLearningConfig.DETAIL_LOG_STEPS:
                    best_q_values = q[state,:]
                    print(f"     决策依据: 选择最大Q值动作")
                    for act_idx, (act_name, q_val) in enumerate(zip(action_names, best_q_values)):
                        marker = " ✅" if act_idx == action else ""
                        print(f"       动作'{act_name}': Q={q_val:.4f}{marker}")
                    sys.stdout.flush()
                
            # 记录动作前的Q值
            old_q_value = q[state, action] if is_training else 0
            
            if show_episode_detail and episode_steps <= QLearningConfig.DETAIL_LOG_STEPS:
                print(f"     ➡️  选择动作: '{action_names[action]}' ({action_type})")
                sys.stdout.flush()

            new_state,reward,terminated,truncated,_ = env.step(action)
            episode_reward += reward

            if is_training:
                # 🎯 奖励塑造: 多层次的奖励系统
                shaped_reward = reward  # 基础奖励
                
                if terminated and reward == 0.0:  # 掉入冰洞
                    shaped_reward += QLearningConfig.HOLE_PENALTY  # 负奖励惩罚掉入冰洞
                elif not terminated:  # 未结束，给予步数惩罚
                    shaped_reward += QLearningConfig.STEP_PENALTY  # 每步小惩罚，鼓励尽快到达
                
                # 计算到目标的距离奖励（可选）
                # 这里可以添加基于距离的奖励，但需要知道目标位置
                
                # Q-learning 更新公式详细过程
                if QLearningConfig.CACHE_Q_MAX and hasattr(env, '_cached_q_max') and env._cached_q_max is not None:
                    max_next_q = env._cached_q_max[new_state]
                else:
                    max_next_q = np.max(q[new_state,:])
                    if QLearningConfig.CACHE_Q_MAX:
                        if not hasattr(env, '_cached_q_max'):
                            env._cached_q_max = np.zeros(env.observation_space.n)
                        env._cached_q_max[new_state] = max_next_q
                
                target = shaped_reward + discount_factor_g * max_next_q
                td_error = target - old_q_value
                new_q_value = old_q_value + learning_rate_a * td_error
                q[state,action] = new_q_value
                
                # 更新缓存
                if QLearningConfig.CACHE_Q_MAX and hasattr(env, '_cached_q_max'):
                    env._cached_q_max[state] = np.max(q[state,:])
                
                # 详细日志输出 - Q值更新过程
                if show_episode_detail and episode_steps <= QLearningConfig.DETAIL_LOG_STEPS:
                    print(f"     🔄 状态转移: {state} → {new_state} (行{new_state//QLearningConfig.MAP_SIZE}, 列{new_state%QLearningConfig.MAP_SIZE})")
                    print(f"     🏆 环境奖励: r = {reward}")
                    if shaped_reward != reward:
                        print(f"     🎯 奖励塑造: shaped_r = {shaped_reward} (掉入冰洞惩罚 {QLearningConfig.HOLE_PENALTY})")
                    else:
                        print(f"     🎯 最终奖励: shaped_r = {shaped_reward}")
                    print(f"     📊 Q-Learning 更新计算详细过程:")
                    print(f"       🔹 步骤1: 获取当前Q值")
                    print(f"         Q(s={state}, a={action}) = {old_q_value:.4f}")
                    print(f"       🔹 步骤2: 查看新状态的所有Q值")
                    print(f"         Q({new_state}, :) = {q[new_state,:]}")
                    print(f"       🔹 步骤3: 找到新状态的最大Q值")
                    print(f"         max Q(s'={new_state}, a') = {max_next_q:.4f}")
                    print(f"       🔹 步骤4: 使用贝尔曼方程计算目标值")
                    print(f"         Target = shaped_r + γ × max Q(s', a')")
                    print(f"         Target = {shaped_reward} + {discount_factor_g} × {max_next_q:.4f}")
                    print(f"         Target = {target:.4f}")
                    print(f"       🔹 步骤5: 计算时序差分(TD)误差")
                    print(f"         TD_error = Target - Q(s,a)")
                    print(f"         TD_error = {target:.4f} - {old_q_value:.4f}")
                    print(f"         TD_error = {td_error:.4f}")
                    print(f"       🔹 步骤6: 使用学习率更新Q值")
                    print(f"         Q_new(s,a) = Q_old(s,a) + α × TD_error")
                    print(f"         Q_new({state},{action}) = {old_q_value:.4f} + {learning_rate_a} × {td_error:.4f}")
                    print(f"         Q_new({state},{action}) = {new_q_value:.4f}")
                    print(f"     📋 更新后的Q表状态:")
                    print(f"       当前状态Q值: Q({state},:) = {q[state,:]}")
                    
                    if terminated:
                        if reward > 0:
                            print(f"     🎉 到达目标! 获得奖励 {reward}")
                        else:
                            print(f"     ❄️ 掉入冰洞! 回合结束")
                    
                    print(f"     {'='*QLearningConfig.LOG_SEPARATOR_LENGTH}")
                    # 每个动作步骤后立即刷新输出
                    sys.stdout.flush()
            else:
                if show_episode_detail and episode_steps <= QLearningConfig.DETAIL_LOG_STEPS:
                    print(f"     🔄 状态转移: {state} → {new_state} (测试模式，不更新Q值)")
                    sys.stdout.flush()

            state = new_state

        epsilon = max(epsilon - epsilon_decay_rate, QLearningConfig.MIN_EPSILON)

        # 改进学习率调整：当epsilon接近0时，逐渐降低学习率
        if epsilon < QLearningConfig.EPSILON_THRESHOLD:
            learning_rate_a = max(learning_rate_a * QLearningConfig.LEARNING_RATE_DECAY, QLearningConfig.MIN_LEARNING_RATE)

        if reward == QLearningConfig.SUCCESS_REWARD:
            rewards_per_episode[i] = QLearningConfig.SUCCESS_REWARD
            
        # 回合总结
        if show_episode_detail:
            success_rate = np.sum(rewards_per_episode[:i+1]) / (i+1) * 100
            print(f"  📋 回合总结: {episode_steps}步, 奖励={episode_reward}, 当前成功率={success_rate:.1f}%")
            
        # 进度更新和checkpoint保存 - 每5%的进度保存一次
        if log_details and is_training and i % max(1, episodes // QLearningConfig.CHECKPOINT_INTERVAL_MAJOR) == 0 and i > 0:
            success_rate = np.sum(rewards_per_episode[:i+1]) / (i+1) * 100
            avg_q = np.mean(np.max(q, axis=1))
            print(f"\n📈 训练进度 {i+1}/{episodes} ({(i+1)/episodes*100:.0f}%): "
                  f"成功率={success_rate:.1f}%, 平均Q值={avg_q:.3f}, ε={epsilon:.4f}")
            
            # 保存checkpoint
            try:
                checkpoint = {
                    'q_table': q,
                    'episode': start_episode + i + 1,  # 修正：使用全局回合数
                    'epsilon': epsilon,
                    'learning_rate': learning_rate_a,
                    'success_rate': success_rate,
                    'avg_q': avg_q
                }
                with open(checkpoint_file, 'wb') as f:
                    pickle.dump(checkpoint, f)
                print(f"💾 已保存checkpoint到 {checkpoint_file}")
            except Exception as e:
                print(f"⚠️  保存checkpoint失败: {e}")
            
            sys.stdout.flush()  # 强制刷新输出
        
        # 每1000回合自动保存checkpoint（即使没有详细日志）
        elif is_training and i % QLearningConfig.CHECKPOINT_INTERVAL_AUTO == 0 and i > 0:
            try:
                checkpoint = {
                    'q_table': q,
                    'episode': start_episode + i + 1,  # 修正：使用全局回合数
                    'epsilon': epsilon,
                    'learning_rate': learning_rate_a,
                    'success_rate': np.sum(rewards_per_episode[:i+1]) / (i+1) * 100,
                    'avg_q': np.mean(np.max(q, axis=1))
                }
                with open(checkpoint_file, 'wb') as f:
                    pickle.dump(checkpoint, f)
                if not log_details:
                    print(f"\r💾 自动保存checkpoint到 {checkpoint_file} (回合 {start_episode + i + 1})", end='', flush=True)
            except Exception as e:
                if not log_details:
                    print(f"\r⚠️  保存checkpoint失败: {e}", end='', flush=True)
        
        # 每100回合快速保存（防止频繁中断）
        elif is_training and i % QLearningConfig.CHECKPOINT_INTERVAL_QUICK == 0 and i > 0:
            try:
                quick_checkpoint = {
                    'q_table': q,
                    'episode': start_episode + i + 1,  # 修正：使用全局回合数
                    'epsilon': epsilon,
                    'learning_rate': learning_rate_a,
                    'success_rate': np.sum(rewards_per_episode[:i+1]) / (i+1) * 100,
                    'avg_q': np.mean(np.max(q, axis=1))
                }
                with open(checkpoint_file + '.tmp', 'wb') as f:
                    pickle.dump(quick_checkpoint, f)
                # 原子性保存：先写临时文件，再重命名
                if os.path.exists(checkpoint_file + '.tmp'):
                    os.rename(checkpoint_file + '.tmp', checkpoint_file)
            except Exception as e:
                # 静默处理快速保存失败
                pass

    env.close()
    
    # 训练完成后换行，让进度条显示完整
    if not log_details and is_training:
        print()  # 换行

    # 训练完成总结
    if log_details and is_training:
        final_success_rate = np.sum(rewards_per_episode) / episodes * 100
        total_steps = step_count
        avg_final_q = np.mean(np.max(q, axis=1))
        
        print("\n" + "=" * QLearningConfig.SEPARATOR_LENGTH)
        print("🏁 训练完成!")
        print(f"📊 最终统计:")
        print(f"   • 总回合数: {episodes}")
        print(f"   • 总步数: {total_steps}")
        print(f"   • 成功率: {final_success_rate:.2f}%")
        print(f"   • 平均Q值: {avg_final_q:.4f}")
        print(f"   • 最终探索率: {epsilon:.4f}")
        print(f"   • 最终学习率: {learning_rate_a:.4f}")
        
        # 显示学到的最优策略示例
        print(f"\n🧠 学到的策略示例 (前16个状态的最优动作):")
        for row in range(QLearningConfig.STRATEGY_DISPLAY_ROWS):
            actions_row = []
            for col in range(QLearningConfig.STRATEGY_DISPLAY_COLS):
                state = row * QLearningConfig.MAP_SIZE + col
                best_action = np.argmax(q[state, :])
                actions_row.append(action_names[best_action])
            print(f"   行{row}: {' '.join(f'{action:^4}' for action in actions_row)}")
        print("=" * QLearningConfig.SEPARATOR_LENGTH)

    sum_rewards = np.zeros(episodes)
    for t in range(episodes):
        sum_rewards[t] = np.sum(rewards_per_episode[max(0, t-QLearningConfig.MOVING_AVERAGE_WINDOW):(t+1)])
    plt.figure(figsize=QLearningConfig.PLOT_FIGURE_SIZE)
    plt.subplot(2, 1, 1)
    plt.plot(sum_rewards)
    plt.title(f'Q-Learning 训练过程 - 移动平均奖励 ({QLearningConfig.MOVING_AVERAGE_WINDOW}回合窗口)', fontsize=14)
    plt.xlabel('回合数')
    plt.ylabel(f'累积成功次数 (过去{QLearningConfig.MOVING_AVERAGE_WINDOW}回合)')
    plt.grid(True, alpha=0.3)
    
    # 添加成功率图
    plt.subplot(2, 1, 2)
    success_rate_curve = np.zeros(episodes)
    for t in range(episodes):
        success_rate_curve[t] = np.sum(rewards_per_episode[:t+1]) / (t+1) * 100
    plt.plot(success_rate_curve, 'r-', linewidth=2)
    plt.title('累积成功率变化', fontsize=14)
    plt.xlabel('回合数')
    plt.ylabel('成功率 (%)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 确保图表可以正确保存
    try:
        plt.savefig(QLearningConfig.PLOT_FILE, dpi=QLearningConfig.PLOT_DPI, bbox_inches='tight')
        if log_details:
            print(f"📈 训练图表已保存到 {QLearningConfig.PLOT_FILE}")
    except Exception as e:
        print(f"⚠️  保存图表失败: {e}")
        # 尝试保存到当前目录
        try:
            plt.savefig('./' + QLearningConfig.PLOT_FILE, dpi=QLearningConfig.PLOT_DPI, bbox_inches='tight')
            if log_details:
                print(f"📈 训练图表已保存到 ./{QLearningConfig.PLOT_FILE}")
        except Exception as e2:
            print(f"❌ 保存图表完全失败: {e2}")
    
    plt.close()  # 关闭图表释放内存

    if is_training:
        with open(QLearningConfig.MODEL_FILE, "wb") as f:
            pickle.dump(q, f)
        if log_details:
            print(f"💾 训练模型已保存到 {QLearningConfig.MODEL_FILE}")

if __name__ == '__main__':
    # 训练模式示例 - 带详细日志
    print("🚀 启动 Q-Learning 算法演示")
    print("💡 提示: render=True 显示可视化界面，render=False 加快训练速度")
    print("💡 提示: log_details=False 可以关闭详细日志")
    print()
    
    # 你可以调整这些参数:
    # episodes: 训练回合数
    # is_training: True=训练模式, False=测试模式(需要先训练)
    # render: True=显示图形界面, False=无图形界面(更快)
    # log_details: True=显示详细日志, False=静默运行
    
    # 🚀 快速训练阶段 (无可视化，快速学习)
    print(f"🚀 开始快速训练 ({QLearningConfig.DEFAULT_TRAINING_EPISODES}回合)...")
    
    # 检查是否有之前的checkpoint
    checkpoint_file = QLearningConfig.CHECKPOINT_FILE
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'rb') as f:
                checkpoint = pickle.load(f)
                print(f"🔍 发现之前的checkpoint: 已训练到第{checkpoint['episode']}回合")
                print(f"   成功率: {checkpoint['success_rate']:.1f}%, 平均Q值: {checkpoint['avg_q']:.4f}")
                
                # 自动继续训练，无需用户选择
                print("✅ 自动继续之前的训练...")
                # 计算剩余回合数
                remaining_episodes = QLearningConfig.DEFAULT_TRAINING_EPISODES - checkpoint['episode']
                if remaining_episodes > 0:
                    print(f"📊 剩余训练回合数: {remaining_episodes}")
                    run(episodes=remaining_episodes, is_training=True, render=False, log_details=False, checkpoint_file=checkpoint_file)
                else:
                    print("🎉 训练已完成！")
        except Exception as e:
            print(f"⚠️  加载checkpoint失败: {e}, 重新开始训练")
            run(episodes=QLearningConfig.DEFAULT_TRAINING_EPISODES, is_training=True, render=False, log_details=False, checkpoint_file=checkpoint_file)
    else:
        print("🆕 开始新训练...")
        run(episodes=QLearningConfig.DEFAULT_TRAINING_EPISODES, is_training=True, render=False, log_details=False, checkpoint_file=checkpoint_file)
    
    print("\n✅ 训练完成！现在开始可视化演示...")
    input("按Enter键开始演示...")
    
    # 🎮 可视化演示阶段 (显示训练后的效果，不限回合数)
    print("\n" + "="*QLearningConfig.DEMO_SEPARATOR_LENGTH)
    print("🎮 演示训练后的智能体表现 (不限回合数，按Ctrl+C停止)")
    try:
        run(episodes=QLearningConfig.DEMO_EPISODES, is_training=False, render=True, log_details=True)
    except KeyboardInterrupt:
        print("\n👋 演示结束")
