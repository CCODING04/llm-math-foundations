"""贝叶斯更新演示：随着观测数据增多，后验如何变化"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'STHeiti']
plt.rcParams['axes.unicode_minus'] = False

# 场景：估计硬币正面概率 p
# 先验：Beta(1, 1) = 均匀分布（什么都不知道）
# 似然：抛硬币 → 二项分布
# 后验：Beta(α + 正面数, β + 反面数）

true_p = 0.3  # 真实正面概率（未知）
np.random.seed(42)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

alpha_prior, beta_prior = 1, 1  # 先验参数
observations = [0, 5, 15, 30, 60, 120]  # 不同观测次数

x = np.linspace(0, 1, 500)

for idx, n_obs in enumerate(observations):
    # 模拟观测
    if n_obs > 0:
        data = np.random.binomial(1, true_p, n_obs)
        heads = data.sum()
        tails = n_obs - heads
    else:
        heads, tails = 0, 0
    
    # 后验参数
    post_alpha = alpha_prior + heads
    post_beta = beta_prior + tails
    
    ax = axes[idx]
    # 绘制先验（灰色虚线）
    ax.plot(x, stats.beta.pdf(x, alpha_prior, beta_prior), '--', color='gray', alpha=0.5, label='先验 Beta(1,1)')
    # 绘制后验
    ax.plot(x, stats.beta.pdf(x, post_alpha, post_beta), '-', color='#E91E63', linewidth=2, 
            label=f'后验 Beta({post_alpha},{post_beta})')
    # 真实值
    ax.axvline(x=true_p, color='#2196F3', linestyle=':', linewidth=2, label=f'真实 p={true_p}')
    
    ax.set_title(f'观测 {n_obs} 次后（正面{heads}次）', fontsize=12)
    ax.legend(fontsize=9)
    ax.set_xlabel('p')
    ax.set_ylabel('密度')

fig.suptitle('贝叶斯更新：随着数据增多，对硬币概率的估计越来越准', fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig('./images/bayesian_update.png', dpi=150, bbox_inches='tight')
print("✅ 贝叶斯更新图已保存")
