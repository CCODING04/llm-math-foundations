"""大数定律演示：掷骰子的样本均值如何趋近期望值"""
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'STHeiti']
plt.rcParams['axes.unicode_minus'] = False

np.random.seed(42)

# 掷骰子实验
max_throws = 10000
rolls = np.random.randint(1, 7, size=max_throws)
cumulative_mean = np.cumsum(rolls) / np.arange(1, max_throws + 1)
expected_value = 3.5

# 绘图
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(range(1, max_throws + 1), cumulative_mean, alpha=0.7, linewidth=0.8, color='#2196F3')
ax.axhline(y=expected_value, color='#F44336', linestyle='--', linewidth=2, label=f'理论期望 E[X] = {expected_value}')
ax.set_xlabel('投掷次数 n', fontsize=12)
ax.set_ylabel('样本均值', fontsize=12)
ax.set_title('大数定律演示：骰子点数的样本均值 → 3.5', fontsize=14)
ax.legend(fontsize=12)
ax.set_xlim(1, max_throws)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('./images/law_of_large_numbers.png', dpi=150)
print("✅ 大数定律图已保存")
