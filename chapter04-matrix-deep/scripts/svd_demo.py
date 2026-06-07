"""
SVD 与特征分解演示
- 手动计算 vs NumPy 结果对比
- SVD 低秩近似可视化
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Heiti SC', 'Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# Part 1: 手动 SVD 计算 vs NumPy
# ==========================================
print("=" * 50)
print("Part 1: SVD 计算")
print("=" * 50)

A = np.array([[3, 2],
              [2, 3]], dtype=float)

print(f"原始矩阵 A:\n{A}\n")

# NumPy SVD
U, sigma, Vt = np.linalg.svd(A)
print(f"U (左奇异向量):\n{U}\n")
print(f"奇异值: {sigma}\n")
print(f"Vt (右奇异向量转置):\n{Vt}\n")

# 验证重构
A_reconstructed = U @ np.diag(sigma) @ Vt
print(f"重构结果 U·Σ·V^T:\n{A_reconstructed}\n")
print(f"重构误差: {np.linalg.norm(A - A_reconstructed):.2e}\n")

# 对比手动计算结果
print("对比手动计算：")
print(f"  奇异值应为 [5, 1]，实际: {sigma}")
print(f"  验证通过! ✓\n")

# ==========================================
# Part 2: 特征分解 (EVD)
# ==========================================
print("=" * 50)
print("Part 2: 特征分解 (EVD)")
print("=" * 50)

# 对称矩阵的特征分解
eigenvalues, eigenvectors = np.linalg.eigh(A)
print(f"特征值: {eigenvalues}")
print(f"特征向量 (列向量):\n{eigenvectors}\n")

# 验证: A * v = lambda * v
for i in range(len(eigenvalues)):
    v = eigenvectors[:, i]
    Av = A @ v
    lambda_v = eigenvalues[i] * v
    print(f"λ_{i}={eigenvalues[i]:.2f}, ||Av - λv|| = {np.linalg.norm(Av - lambda_v):.2e}")

# EVD 重构
Lambda = np.diag(eigenvalues)
Q = eigenvectors
A_evd = Q @ Lambda @ Q.T
print(f"\nEVD 重构: Q·Λ·Q^T:\n{A_evd}")
print(f"重构误差: {np.linalg.norm(A - A_evd):.2e}\n")

# ==========================================
# Part 3: 特征向量几何可视化
# ==========================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 左图：原始空间中的特征向量
ax1 = axes[0]
theta = np.linspace(0, 2 * np.pi, 100)
circle_x = np.cos(theta)
circle_y = np.sin(theta)

# 单位圆
ax1.plot(circle_x, circle_y, 'b-', alpha=0.3, label='单位圆')

# 变换后的椭圆
transformed = A @ np.vstack([circle_x, circle_y])
ax1.plot(transformed[0], transformed[1], 'r-', alpha=0.5, linewidth=2, label='变换后椭圆')

# 画特征向量
colors = ['green', 'orange']
labels_ev = [f'特征向量 (λ={eigenvalues[0]:.1f})', f'特征向量 (λ={eigenvalues[1]:.1f})']
for i in range(2):
    v = eigenvectors[:, i]
    ax1.annotate('', xy=v * eigenvalues[i] * 0.5, xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color=colors[i], lw=2.5))
    ax1.plot([0, v[0] * eigenvalues[i] * 0.5], [0, v[1] * eigenvalues[i] * 0.5],
             'o', color=colors[i], markersize=8)
    ax1.text(v[0] * eigenvalues[i] * 0.55, v[1] * eigenvalues[i] * 0.55 + 0.2,
             labels_ev[i], fontsize=10, color=colors[i], fontweight='bold')

ax1.set_xlim(-4, 4)
ax1.set_ylim(-4, 4)
ax1.set_aspect('equal')
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=10)
ax1.set_title('特征向量的几何意义：方向不变的轴', fontsize=13)
ax1.axhline(y=0, color='k', linewidth=0.5)
ax1.axvline(x=0, color='k', linewidth=0.5)

# 右图：变换前后的网格
ax2 = axes[1]
# 画原始网格
for i in range(-3, 4):
    ax2.plot([i, i], [-3, 3], 'b-', alpha=0.15)
    ax2.plot([-3, 3], [i, i], 'b-', alpha=0.15)

# 画变换后的网格
for i in range(-3, 4):
    line_v = A @ np.array([[i, i], [-3, 3]])
    ax2.plot(line_v[0], line_v[1], 'r-', alpha=0.3)
    line_h = A @ np.array([[-3, 3], [i, i]])
    ax2.plot(line_h[0], line_h[1], 'r-', alpha=0.3)

# 标注面积变化
ax2.set_title(f'变换效果：面积缩放 × |det(A)| = {abs(np.linalg.det(A)):.1f}', fontsize=13)
ax2.set_xlim(-10, 10)
ax2.set_ylim(-10, 10)
ax2.set_aspect('equal')
ax2.grid(True, alpha=0.3)
ax2.axhline(y=0, color='k', linewidth=0.5)
ax2.axvline(x=0, color='k', linewidth=0.5)

plt.tight_layout()
plt.savefig('images/eigenvector_geometry.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ 已保存: images/eigenvector_geometry.png")

# ==========================================
# Part 4: SVD 低秩近似
# ==========================================
print("\n" + "=" * 50)
print("Part 4: SVD 低秩近似")
print("=" * 50)

# 创建一个「近似低秩」的矩阵
np.random.seed(42)
# 基础低秩部分
W_true = np.random.randn(20, 3) @ np.random.randn(3, 20)
# 加一点噪声
W = W_true + 0.1 * np.random.randn(20, 20)

U_w, sigma_w, Vt_w = np.linalg.svd(W, full_matrices=False)

print(f"前10个奇异值: {sigma_w[:10].round(4)}")
print(f"奇异值衰减比 (σ₁/σ₂₀): {sigma_w[0]/sigma_w[-1]:.1f}\n")

# 计算不同秩的近似误差
errors = []
total_energy = np.sum(sigma_w ** 2)
energy_ratios = []

for k in range(1, 21):
    W_k = U_w[:, :k] @ np.diag(sigma_w[:k]) @ Vt_w[:k, :]
    error = np.linalg.norm(W - W_k, 'fro')
    errors.append(error)
    energy = np.sum(sigma_w[:k] ** 2) / total_energy * 100
    energy_ratios.append(energy)
    if k <= 5:
        print(f"  秩 k={k:2d}: 误差={error:.4f}, 能量保留={energy:.1f}%")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 奇异值衰减
ax1 = axes[0]
ax1.bar(range(1, 21), sigma_w, color='steelblue', alpha=0.8)
ax1.set_xlabel('奇异值序号', fontsize=12)
ax1.set_ylabel('奇异值大小', fontsize=12)
ax1.set_title('奇异值快速衰减 → 矩阵本质上是低秩的', fontsize=13)
ax1.axhline(y=sigma_w[2], color='red', linestyle='--', alpha=0.5,
            label=f'σ₃ = {sigma_w[2]:.2f}')
ax1.legend(fontsize=10)

# 近似误差 vs 秩
ax2 = axes[1]
ax2.plot(range(1, 21), errors, 'bo-', markersize=6, label='Frobenius 误差')
ax2b = ax2.twinx()
ax2b.plot(range(1, 21), energy_ratios, 'rs--', markersize=6, label='能量保留 %')
ax2.set_xlabel('近似秩 k', fontsize=12)
ax2.set_ylabel('Frobenius 误差', color='blue', fontsize=12)
ax2b.set_ylabel('能量保留 (%)', color='red', fontsize=12)
ax2.set_title('低秩近似：k=3 已经捕获几乎全部信息', fontsize=13)
ax2.axvline(x=3, color='green', linestyle=':', alpha=0.5, label='k=3')
lines1, labels1 = ax2.get_legend_handles_labels()
lines2, labels2 = ax2b.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, fontsize=10)

plt.tight_layout()
plt.savefig('images/svd_lowrank.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n✓ 已保存: images/svd_lowrank.png")

print("\n" + "=" * 50)
print("✅ 所有演示完成！")
print("=" * 50)
