import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import time
import sys

def run_pid_control(episodes=10, render=True, save_data=True, p=0.1, i=0.0001, d=0.005, experiment_title="PID控制实验"):
    """
    使用PID控制器控制CartPole平衡
    
    参数说明:
    - episodes: 运行回合数
    - render: 是否显示可视化界面
    - save_data: 是否保存性能数据
    - p: 比例参数 - 控制角度偏差的响应强度
    - i: 积分参数 - 控制位置偏差的累积响应
    - d: 微分参数 - 控制角速度的响应
    - experiment_title: 实验主题标题
    """
    
    # 参数验证
    if episodes <= 0:
        print("❌ 回合数必须大于0")
        return [], []
    
    if not isinstance(episodes, int):
        print("❌ 回合数必须是整数")
        return [], []
    
    # 创建环境
    try:
        # 尝试使用CartPole-v0，它通常有更高的步数限制
        env = gym.make('CartPole-v0', render_mode='human' if render else None)
        print("✅ 使用CartPole-v0环境 (更高步数限制)")
    except:
        try:
            # 如果v0不可用，使用v1但尝试修改参数
            env = gym.make('CartPole-v1', render_mode='human' if render else None)
            print("✅ 使用CartPole-v1环境")
            print("⚠️  注意: v1环境有500步限制，如果达到限制会提前结束")
        except Exception as e:
            print(f"❌ 创建环境失败: {e}")
            print("请确保已安装gymnasium和CartPole环境")
            return [], []
    
    # 记录数据
    episode_durations = []  # 每回合持续时间
    episode_errors = []     # 每回合的平均误差
    
    print("🎯 CartPole PID控制演示")
    print(f"📊 PID参数: P={p}, I={i}, D={d}")
    print(f"🎮 运行 {episodes} 个回合")
    print("=" * 60)
    
    # 在环境中显示实验信息
    if render:
        # 等待环境渲染窗口出现
        time.sleep(0.5)
        # 尝试通过pygame设置窗口标题来显示实验信息
        try:
            import pygame
            # 设置窗口标题显示实验信息
            pygame.display.set_caption(f"CartPole PID控制 - {experiment_title} | P={p}, I={i}, D={d}")
        except:
            # 如果pygame不可用，至少打印信息
            pass
        
        print(f"\n🎬 当前实验: {experiment_title}")
        print(f"📊 当前PID参数: P={p}, I={i}, D={d}")
        print("💡 观察杆子的平衡状态和控制效果")
        print("💡 注意查看窗口标题栏的实验信息")
    
    for episode in range(episodes):
        observation, _ = env.reset()
        episode_duration = 0
        episode_error_sum = 0
        
        # 记录控制信号的历史（限制大小防止内存泄漏）
        control_signals = []
        angle_errors = []
        
        print(f"\n📍 回合 {episode + 1}/{episodes}")
        
        while True:  # 无步数限制，运行到杆子落下为止
            if render:
                env.render()
                time.sleep(0.01)  # 控制显示速度
            
            # 提取观测值
            # observation[0]: 小车位置 (cart position)
            # observation[1]: 小车速度 (cart velocity) 
            # observation[2]: 杆子角度 (pole angle) - 这是我们要控制的目标
            # observation[3]: 杆子角速度 (pole angular velocity)
            cart_pos = observation[0]
            cart_vel = observation[1] 
            pole_angle = observation[2]  # 角度误差（目标角度为0）
            pole_angular_vel = observation[3]
            
            # 计算控制误差
            angle_error = abs(pole_angle)  # 角度误差的绝对值
            episode_error_sum += angle_error
            angle_errors.append(angle_error)
            
            # PID控制算法
            # 控制信号 = P*角度 + I*位置 + D*角速度 + 位置反馈
            control_signal = (pole_angle * p + 
                            cart_pos * i + 
                            pole_angular_vel * d + 
                            cart_vel * i)
            
            # 根据控制信号决定动作
            if control_signal > 0:
                action = 1  # 向右推
            else:
                action = 0  # 向左推
            
            # 记录数据用于分析（限制历史记录大小）
            control_signals.append(control_signal)
            
            # 执行动作
            observation, reward, terminated, truncated, _ = env.step(action)
            episode_duration += 1
            
            # 每100步显示一次状态信息
            if episode_duration % 100 == 0 and episode_duration > 0:
                print(f"  持续时间: {episode_duration}步, 角度={pole_angle:.4f}, 位置={cart_pos:.4f}, "
                      f"控制信号={control_signal:.4f}")
                print("  💡 观察杆子平衡状态，杆子倒下后会自动进入下一阶段")
                sys.stdout.flush()
            
            # 检查是否结束
            if terminated:
                print(f"  🔴 回合结束原因: 杆子倒下 (terminated=True)")
                break
            elif truncated:
                print(f"  🟡 回合结束原因: 达到环境最大步数限制 (truncated=True, 步数={episode_duration})")
                print(f"  💡 这是环境限制，不是杆子倒下。如果想继续观察，可以重新运行")
                break
        
        # 回合总结
        episode_durations.append(episode_duration)
        avg_error = episode_error_sum / episode_duration if episode_duration > 0 else 0
        episode_errors.append(avg_error)
        
        print(f"  📋 回合结束: 持续{episode_duration}步, 平均角度误差={avg_error:.4f}")
        
        # 分析控制效果
        if len(control_signals) > 0:
            avg_control = np.mean(np.abs(control_signals))
            max_control = np.max(np.abs(control_signals))
            print(f"  📊 控制信号: 平均幅度={avg_control:.4f}, 最大幅度={max_control:.4f}")
    
    env.close()
    
    # 转换为numpy数组以确保数据类型一致
    episode_durations = np.array(episode_durations)
    episode_errors = np.array(episode_errors)
    
    # 统计结果
    avg_duration = np.mean(episode_durations)
    avg_error = np.mean(episode_errors)
    success_rate = np.sum(episode_durations >= 500) / episodes * 100  # 500步认为成功
    
    print("\n" + "=" * 60)
    print("🏁 PID控制结果统计:")
    print(f"   • 平均持续时间: {avg_duration:.2f}步")
    print(f"   • 平均角度误差: {avg_error:.4f}弧度")
    print(f"   • 成功率: {success_rate:.1f}% (≥500步)")
    print(f"   • 最好成绩: {max(episode_durations)}步")
    print(f"   • PID参数: P={p}, I={i}, D={d}")
    print("=" * 60)
    
    # 保存和可视化数据
    if save_data:
        save_pid_results(episode_durations, episode_errors, p, i, d)
    
    return episode_durations, episode_errors

def save_pid_results(durations, errors, p, i, d):
    """保存PID控制结果并生成图表"""
    
    # 创建图表
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # 持续时间曲线
    ax1.plot(durations, 'b-', linewidth=2, label='每回合持续时间')
    ax1.axhline(y=np.mean(durations), color='r', linestyle='--', 
                label=f'平均持续时间: {np.mean(durations):.1f}步')
    ax1.set_title(f'CartPole PID控制 - 持续时间曲线 (P={p}, I={i}, D={d})', fontsize=14)
    ax1.set_xlabel('回合数')
    ax1.set_ylabel('步数')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 平均误差曲线
    ax2.plot(errors, 'g-', linewidth=2, label='每回合平均误差')
    ax2.axhline(y=np.mean(errors), color='r', linestyle='--', 
                label=f'平均误差: {np.mean(errors):.4f}弧度')
    ax2.axhline(y=0.05, color='orange', linestyle=':', 
                label='目标误差: 0.05弧度') # 假设0.05弧度是目标误差
    ax2.set_title('平均误差曲线', fontsize=14)
    ax2.set_xlabel('回合数')
    ax2.set_ylabel('弧度')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 添加错误处理确保图表可以正确保存
    try:
        plt.savefig('cartpole_pid_results.png', dpi=150, bbox_inches='tight')
        print(f"📈 结果图表已保存到 cartpole_pid_results.png")
    except Exception as e:
        print(f"⚠️  保存图表失败: {e}")
        # 尝试保存到当前目录
        try:
            plt.savefig('./cartpole_pid_results.png', dpi=150, bbox_inches='tight')
            print(f"📈 结果图表已保存到 ./cartpole_pid_results.png")
        except Exception as e2:
            print(f"❌ 保存图表完全失败: {e2}")
    
    plt.close()  # 关闭图表释放内存

def tune_pid_parameters():
    """PID参数调优实验"""
    
    print("🔧 PID参数调优实验")
    print("=" * 50)
    
    # 测试不同的PID参数组合
    param_sets = [
        {'p': 0.1, 'i': 0.0001, 'd': 0.005, 'name': '参考参数'},
        {'p': 0.15, 'i': 0.0001, 'd': 0.005, 'name': '增大P'},
        {'p': 0.1, 'i': 0.0005, 'd': 0.005, 'name': '增大I'}, 
        {'p': 0.1, 'i': 0.0001, 'd': 0.01, 'name': '增大D'},
        {'p': 0.05, 'i': 0.0001, 'd': 0.005, 'name': '减小P'},
    ]
    
    results = []
    
    for params in param_sets:
        print(f"\n🧪 测试 {params['name']}: P={params['p']}, I={params['i']}, D={params['d']}")
        
        # 运行测试 - 现在可以真正测试不同的PID参数
        try:
            durations, errors = run_pid_control(
                episodes=3,  # 减少回合数以加快测试
                render=False, 
                save_data=False,
                p=params['p'], 
                i=params['i'], 
                d=params['d']
            )
            
            if len(durations) > 0 and len(errors) > 0:
                avg_duration = np.mean(durations)
                avg_error = np.mean(errors)
            else:
                avg_duration = 0
                avg_error = float('inf')
                
        except Exception as e:
            print(f"  ⚠️  测试失败: {e}")
            avg_duration = 0
            avg_error = float('inf')
        
        results.append({
            'name': params['name'],
            'params': params,
            'avg_duration': avg_duration,
            'avg_error': avg_error
        })
        
        print(f"  结果: 平均持续时间={avg_duration:.1f}步, 平均误差={avg_error:.4f}弧度")
    
    # 显示最佳参数
    valid_results = [r for r in results if r['avg_error'] != float('inf')]
    if valid_results:
        best_result = min(valid_results, key=lambda x: x['avg_error']) # 误差越小越好
        print(f"\n🏆 最佳参数组合: {best_result['name']}")
        print(f"   参数: P={best_result['params']['p']}, I={best_result['params']['i']}, D={best_result['params']['d']}")
        print(f"   性能: 平均持续时间={best_result['avg_duration']:.1f}步, 平均误差={best_result['avg_error']:.4f}弧度")
    else:
        print("\n⚠️  所有参数组合测试都失败了")

if __name__ == '__main__':
    print("🚀 CartPole PID控制演示")
    print("💡 提示: PID控制器通过比例、积分、微分参数实现杆子平衡控制")
    print("💡 提示: 相比Q-learning，PID控制不需要训练，直接基于物理反馈")
    print()
    
    # 渐进式PID参数演示 - 先单独测试，再组合
    print("🎯 渐进式PID参数演示 - 先单独测试每个参数，再展示组合效果")
    print("=" * 70)
    
    # 阶段1: 所有参数都为0 - 无控制
    print("\n📍 阶段1: 无控制 (P=0, I=0, D=0)")
    print("预期: 杆子会快速倒下，无法保持平衡")
    print("💡 观察杆子如何快速倒下，然后按Enter键继续")
    run_pid_control(episodes=1, render=True, save_data=False, p=0, i=0, d=0, 
                    experiment_title="阶段1: 无控制 - 观察杆子自然倒下")
    
    input("\n按Enter键继续下一阶段...")
    
    # 阶段2: 只启用比例控制P
    print("\n📍 阶段2: 仅比例控制 (P=0.1, I=0, D=0)")
    print("预期: 杆子会有振荡，但能保持一定平衡")
    print("💡 观察振荡现象，然后按Enter键继续")
    run_pid_control(episodes=1, render=True, save_data=False, p=0.1, i=0, d=0,
                    experiment_title="阶段2: 仅比例控制 - 观察振荡现象")
    
    input("\n按Enter键继续下一阶段...")
    
    # 阶段3: 只启用积分控制I
    print("\n📍 阶段3: 仅积分控制 (P=0, I=0.0001, D=0)")
    print("预期: 响应较慢，可能有稳态误差")
    print("💡 观察响应延迟，然后按Enter键继续")
    run_pid_control(episodes=1, render=True, save_data=False, p=0, i=0.0001, d=0,
                    experiment_title="阶段3: 仅积分控制 - 观察响应延迟")
    
    input("\n按Enter键继续下一阶段...")
    
    # 阶段4: 只启用微分控制D
    print("\n📍 阶段4: 仅微分控制 (P=0, I=0, D=0.005)")
    print("预期: 对变化敏感，但单独使用效果有限")
    print("💡 观察变化敏感性，然后按Enter键继续")
    run_pid_control(episodes=1, render=True, save_data=False, p=0, i=0, d=0.005,
                    experiment_title="阶段4: 仅微分控制 - 观察变化敏感性")
    
    input("\n按Enter键继续下一阶段...")
    
    # 阶段5: 比例+微分控制 (PD控制)
    print("\n📍 阶段5: 比例+微分控制 (P=0.1, I=0, D=0.005)")
    print("预期: 振荡减少，控制更稳定")
    print("💡 观察稳定性改善，然后按Enter键继续")
    run_pid_control(episodes=1, render=True, save_data=False, p=0.1, i=0, d=0.005,
                    experiment_title="阶段5: PD控制 - 观察稳定性改善")
    
    input("\n按Enter键继续下一阶段...")
    
    # 阶段6: 比例+积分控制 (PI控制)
    print("\n📍 阶段6: 比例+积分控制 (P=0.1, I=0.0001, D=0)")
    print("预期: 消除稳态误差，但可能增加响应时间")
    run_pid_control(episodes=1, render=True, save_data=False, p=0.1, i=0.0001, d=0,
                    experiment_title="阶段6: PI控制 - 观察稳态误差消除")
    
    input("\n按Enter键继续下一阶段...")
    
    # 阶段7: 完整PID控制
    print("\n📍 阶段7: 完整PID控制 (P=0.1, I=0.0001, D=0.005)")
    print("预期: 最佳控制效果，杆子保持稳定平衡")
    run_pid_control(episodes=3, render=True, save_data=True, p=0.1, i=0.0001, d=0.005,
                    experiment_title="阶段7: 完整PID控制 - 观察最佳平衡效果")
    
    # 参数调优实验
    print("\n" + "="*70)
    print("📍 PID参数调优实验:")
    tune_pid_parameters()
    
    print("\n🎉 演示完成！")
    print("💡 通过这个演示，你可以看到:")
    print("   • P参数(比例): 单独使用时提供基本控制，但可能引起振荡")
    print("   • I参数(积分): 单独使用时响应较慢，主要用于消除稳态误差")
    print("   • D参数(微分): 单独使用时对变化敏感，主要用于减少振荡")
    print("   • PD组合: 提供稳定控制，适合大多数应用")
    print("   • PI组合: 消除稳态误差，但响应可能较慢")
    print("   • PID组合: 综合了所有优点，提供最佳控制性能")
