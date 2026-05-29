"""
模拟策略梯度优化过程
场景：一个简化的"猜数字"游戏，策略从均匀分布逐渐学到正确答案
"""

import numpy as np
import matplotlib.pyplot as plt
import os

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'PingFang SC']
plt.rcParams['axes.unicode_minus'] = False

np.random.seed(42)

# ========== 环境设置 ==========
n_actions = 10
optimal_action = 7
reward_table = np.array([0.1, 0.2, 0.15, 0.3, 0.25, 0.4, 0.5, 1.0, 0.6, 0.35])

# ========== 策略初始化（均匀分布）==========
theta = np.zeros(n_actions)
learning_rate = 0.5
n_episodes = 200
n_steps_per_update = 10

episode_rewards = []
policy_history = []

for episode in range(n_episodes):
    probs = np.exp(theta) / np.sum(np.exp(theta))
    trajectories = np.random.choice(n_actions, size=n_steps_per_update, p=probs)
    rewards = reward_table[trajectories]

    baseline = np.mean(rewards)

    grad = np.zeros(n_actions)
    for a, r in zip(trajectories, rewards):
        one_hot = np.zeros(n_actions)
        one_hot[a] = 1
        grad += (one_hot - probs) * (r - baseline)
    grad /= n_steps_per_update

    theta += learning_rate * grad

    episode_rewards.append(np.mean(rewards))
    if episode % 20 == 0 or episode == n_episodes - 1:
        policy_history.append((episode, probs.copy()))

# ========== 可视化 ==========
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(episode_rewards, alpha=0.3, color='lightblue')
window = 20
smoothed = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
axes[0].plot(range(window-1, len(episode_rewards)), smoothed, color='steelblue', linewidth=2, label=f'滑动平均 (window={window})')
axes[0].axhline(y=reward_table[optimal_action], color='red', linestyle='--', alpha=0.5, label=f'最优奖励 = {reward_table[optimal_action]}')
axes[0].set_xlabel('Episode', fontsize=12)
axes[0].set_ylabel('平均奖励', fontsize=12)
axes[0].set_title('REINFORCE 学习曲线', fontsize=14)
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)

x = np.arange(n_actions)
colors = plt.cm.Blues(np.linspace(0.3, 1.0, len(policy_history)))
for i, (ep, probs) in enumerate(policy_history):
    axes[1].bar(x + i * 0.08 - 0.2, probs, width=0.08, color=colors[i],
                label=f'Episode {ep}', alpha=0.8)
axes[1].axvline(x=optimal_action, color='red', linestyle='--', alpha=0.5, label=f'最优动作 = {optimal_action}')
axes[1].set_xlabel('动作', fontsize=12)
axes[1].set_ylabel('概率', fontsize=12)
axes[1].set_title('策略概率的演化', fontsize=14)
axes[1].set_xticks(x)
axes[1].legend(fontsize=8, ncol=2)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'images')
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, 'policy_gradient.png')
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"图片已保存到 {out_path}")
print(f"\n最终策略：")
final_probs = np.exp(theta) / np.sum(np.exp(theta))
for a in range(n_actions):
    bar = '█' * int(final_probs[a] * 50)
    print(f"  动作 {a}: {final_probs[a]:.3f} {bar}")
