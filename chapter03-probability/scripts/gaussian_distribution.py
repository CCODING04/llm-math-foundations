"""绘制高斯分布"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'STHeiti']
plt.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(figsize=(10, 6))

x = np.linspace(-5, 5, 300)

# 不同 μ 和 σ 的组合
configs = [
    (0, 1, '#2196F3', 'μ=0, σ=1（标准正态）'),
    (0, 0.5, '#4CAF50', 'μ=0, σ=0.5'),
    (0, 2, '#FF9800', 'μ=0, σ=2'),
    (1, 1, '#E91E63', 'μ=1, σ=1'),
]

for mu, sigma, color, label in configs:
    y = stats.norm.pdf(x, mu, sigma)
    ax.plot(x, y, linewidth=2.5, color=color, label=label)

ax.set_xlabel('x', fontsize=13)
ax.set_ylabel('f(x)', fontsize=13)
ax.set_title('高斯（正态）分布：不同参数的效果', fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('./images/gaussian_distribution.png', dpi=150)
print("✅ 高斯分布图已保存")
