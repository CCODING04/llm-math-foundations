"""
用最小二乘法拟合 Kaplan 缩放定律
验证：Loss ∝ N^{-α}
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# 支持中文
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'PingFang SC']
plt.rcParams['axes.unicode_minus'] = False

np.random.seed(42)

# ========== 模拟数据 ==========
N_values = np.array([1e6, 5e6, 1e7, 5e7, 1e8, 5e8, 1e9, 5e9, 1e10, 5e10, 1e11])
true_alpha = 0.076
true_A = 5.0

L_true = true_A * N_values ** (-true_alpha)
noise = np.random.normal(0, 0.02, len(N_values))
L_observed = L_true + noise * L_true

# ========== 最小二乘拟合（双对数空间）==========
log_N = np.log(N_values)
log_L = np.log(L_observed)

log_N_mean = np.mean(log_N)
log_L_mean = np.mean(log_L)

alpha_hat = np.sum((log_N - log_N_mean) * (log_L - log_L_mean)) / np.sum((log_N - log_N_mean) ** 2)
log_A_hat = log_L_mean - alpha_hat * log_N_mean
A_hat = np.exp(log_A_hat)

print(f"真实参数: α = {true_alpha}, A = {true_A}")
print(f"拟合参数: α = {-alpha_hat:.4f}, A = {A_hat:.4f}")
print(f"相对误差: α 误差 = {abs(-alpha_hat - true_alpha)/true_alpha*100:.2f}%")

# ========== 可视化 ==========
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

L_fitted = A_hat * N_values ** alpha_hat
axes[0].scatter(N_values, L_observed, color='steelblue', s=60, zorder=5, label='观测数据')
axes[0].plot(N_values, L_fitted, 'r-', linewidth=2, label=f'拟合: L = {A_hat:.2f} × N^({alpha_hat:.4f})')
axes[0].set_xlabel('模型参数量 N', fontsize=12)
axes[0].set_ylabel('交叉熵损失 L', fontsize=12)
axes[0].set_title('缩放定律（普通坐标）', fontsize=14)
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)

axes[1].scatter(log_N, log_L, color='steelblue', s=60, zorder=5, label='观测数据（取对数）')
axes[1].plot(log_N, log_A_hat + alpha_hat * log_N, 'r-', linewidth=2,
             label=f'线性拟合: ln L = {log_A_hat:.2f} + ({alpha_hat:.4f}) × ln N')
axes[1].set_xlabel('ln(模型参数量 N)', fontsize=12)
axes[1].set_ylabel('ln(交叉熵损失 L)', fontsize=12)
axes[1].set_title('缩放定律（双对数坐标 → 线性）', fontsize=14)
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'images')
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, 'power_law_fit.png')
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"\n图片已保存到 {out_path}")
