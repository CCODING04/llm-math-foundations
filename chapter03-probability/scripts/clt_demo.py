"""中心极限定理演示：任何分布的样本均值都趋向正态分布"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'STHeiti']
plt.rcParams['axes.unicode_minus'] = False

np.random.seed(42)

# 用均匀分布（明显不是正态）来演示 CLT
sample_sizes = [1, 2, 5, 30]
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for idx, n in enumerate(sample_sizes):
    # 每次取 n 个均匀分布样本，计算均值，重复 10000 次
    sample_means = [np.mean(np.random.uniform(0, 1, n)) for _ in range(10000)]
    
    ax = axes[idx]
    ax.hist(sample_means, bins=50, density=True, alpha=0.7, color='#4CAF50', edgecolor='white')
    
    # 叠加理论正态分布
    mu, sigma = 0.5, np.sqrt(1/12) / np.sqrt(n)
    x = np.linspace(mu - 4*sigma, mu + 4*sigma, 200)
    ax.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2, label='理论正态')
    
    ax.set_title(f'样本量 n = {n}', fontsize=13)
    ax.legend(fontsize=10)
    ax.set_xlabel('样本均值')
    ax.set_ylabel('密度')

fig.suptitle('中心极限定理：均匀分布的样本均值 → 正态分布', fontsize=15, y=1.01)
plt.tight_layout()
plt.savefig('./images/clt_demo.png', dpi=150, bbox_inches='tight')
print("✅ CLT 演示图已保存")
