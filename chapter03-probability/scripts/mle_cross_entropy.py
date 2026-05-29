"""演示 MLE 等价于最小化交叉熵"""
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'STHeiti']
plt.rcParams['axes.unicode_minus'] = False

np.random.seed(42)

# 模拟：真实硬币概率 p_true = 0.3，抛 100 次
p_true = 0.3
n = 100
data = np.random.binomial(1, p_true, n)
k = data.sum()  # 正面次数

# 对不同的 p 值计算：对数似然 和 交叉熵
p_range = np.linspace(0.01, 0.99, 200)

# 对数似然：ℓ(p) = k*log(p) + (n-k)*log(1-p)
log_likelihood = k * np.log(p_range) + (n - k) * np.log(1 - p_range)

# 交叉熵：H = -ℓ(p)/n
cross_entropy = -log_likelihood / n

# MLE 估计
p_mle = k / n

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# 左图：对数似然
ax1.plot(p_range, log_likelihood, color='#2196F3', linewidth=2)
ax1.axvline(x=p_mle, color='#F44336', linestyle='--', linewidth=2, label=f'MLE: p = {p_mle:.2f}')
ax1.axvline(x=p_true, color='#4CAF50', linestyle=':', linewidth=2, label=f'真实: p = {p_true}')
ax1.set_xlabel('参数 p', fontsize=12)
ax1.set_ylabel('对数似然 ℓ(p)', fontsize=12)
ax1.set_title('极大似然估计：找最大值', fontsize=14)
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)

# 右图：交叉熵
ax2.plot(p_range, cross_entropy, color='#FF9800', linewidth=2)
ax2.axvline(x=p_mle, color='#F44336', linestyle='--', linewidth=2, label=f'MLE: p = {p_mle:.2f}')
ax2.axvline(x=p_true, color='#4CAF50', linestyle=':', linewidth=2, label=f'真实: p = {p_true}')
ax2.set_xlabel('参数 p', fontsize=12)
ax2.set_ylabel('交叉熵 H(p)', fontsize=12)
ax2.set_title('交叉熵损失：找最小值', fontsize=14)
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)

fig.suptitle('MLE = 最小化交叉熵：同一个问题的两面', fontsize=15, y=1.02)
plt.tight_layout()
plt.savefig('./images/mle_cross_entropy.png', dpi=150, bbox_inches='tight')
print("✅ MLE 与交叉熵对比图已保存")
print(f"   MLE 估计: p = {p_mle:.2f}，真实值: p = {p_true}")
