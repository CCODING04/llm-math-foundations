"""
可视化 PPO 的裁剪函数
展示为什么裁剪能防止策略更新过大
"""

import numpy as np
import matplotlib.pyplot as plt

epsilon = 0.2
ratio = np.linspace(0.0, 2.0, 500)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 左图：不同优势值下的 PPO 目标
for A_val, color, label in [(1.0, 'steelblue', 'A > 0 (好动作)'),
                              (-1.0, 'coral', 'A < 0 (坏动作)')]:
    # 未裁剪
    unclipped = ratio * A_val
    # 裁剪后
    clipped_ratio = np.clip(ratio, 1 - epsilon, 1 + epsilon)
    clipped = clipped_ratio * A_val
    # PPO 取 min
    ppo_obj = np.minimum(unclipped, clipped)
    
    axes[0].plot(ratio, ppo_obj, color=color, linewidth=2.5, label=label)

axes[0].axhline(y=0, color='gray', linestyle='-', alpha=0.3)
axes[0].axvline(x=1.0, color='gray', linestyle='--', alpha=0.3)
axes[0].axvline(x=1-epsilon, color='green', linestyle=':', alpha=0.5, label=f'1-ε = {1-epsilon}')
axes[0].axvline(x=1+epsilon, color='green', linestyle=':', alpha=0.5, label=f'1+ε = {1+epsilon}')
axes[0].set_xlabel('概率比 r(θ) = π_θ / π_old', fontsize=12)
axes[0].set_ylabel('PPO 目标函数', fontsize=12)
axes[0].set_title(f'PPO-Clip 目标函数 (ε = {epsilon})', fontsize=14)
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)
axes[0].set_xlim(0, 2)

# 右图：DPO 损失函数
beta = 0.5
margin = np.linspace(-5, 5, 500)  # log π_w/π_ref - log π_l/π_ref
dpo_loss = -np.log(1 / (1 + np.exp(-beta * margin)))  # -log σ(β * margin)

axes[1].plot(margin, dpo_loss, color='steelblue', linewidth=2.5)
axes[1].axhline(y=0, color='gray', linestyle='-', alpha=0.3)
axes[1].axvline(x=0, color='gray', linestyle='--', alpha=0.3)
axes[1].annotate('模型偏好与人类一致\n(margin > 0, 损失 → 0)', 
                xy=(3, 0.1), fontsize=10, color='green',
                ha='center')
axes[1].annotate('模型偏好与人类相反\n(margin < 0, 损失 → 大)', 
                xy=(-3, 2.5), fontsize=10, color='red',
                ha='center')
axes[1].set_xlabel('β × (log π_θ(y_w)/π_ref(y_w) - log π_θ(y_l)/π_ref(y_l))', fontsize=10)
axes[1].set_ylabel('DPO 损失', fontsize=12)
axes[1].set_title(f'DPO 损失函数 (β = {beta})', fontsize=14)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('./images/ppo_dpo_visual.png', dpi=150, bbox_inches='tight')
print("图片已保存到 ./images/ppo_dpo_visual.png")
