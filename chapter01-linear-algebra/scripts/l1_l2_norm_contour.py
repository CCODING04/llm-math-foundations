"""
L1 vs L2 范数的等高线对比
展示两种范数的几何形状差异，解释 L1 为什么能产生稀疏解
"""
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['PingFang SC', 'Heiti SC', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False
import numpy as np

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 创建网格
delta = 0.025
x = np.arange(-1.5, 1.5, delta)
y = np.arange(-1.5, 1.5, delta)
X, Y = np.meshgrid(x, y)

# L2 范数
L2 = np.sqrt(X**2 + Y**2)

# L1 范数
L1 = np.abs(X) + np.abs(Y)

# 绘制 L2 等高线
ax = axes[0]
cs = ax.contour(X, Y, L2, levels=[0.25, 0.5, 0.75, 1.0, 1.25], cmap='Blues')
ax.clabel(cs, inline=True, fontsize=9, fmt='%.2f')
ax.plot(np.cos(np.linspace(0, 2*np.pi, 100)), np.sin(np.linspace(0, 2*np.pi, 100)),
        'b-', lw=2, label='‖v‖₂ = 1')
ax.set_title('L2 范数等高线（圆形）\n正则化效果：温和地缩小所有权重', fontsize=12, fontweight='bold')
ax.set_xlabel('w₁', fontsize=12)
ax.set_ylabel('w₂', fontsize=12)
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=10)

# 绘制 L1 等高线
ax = axes[1]
cs = ax.contour(X, Y, L1, levels=[0.25, 0.5, 0.75, 1.0, 1.25], cmap='Oranges')
ax.clabel(cs, inline=True, fontsize=9, fmt='%.2f')
# 画 L1 单位球（菱形）
diamond_x = [1, 0, -1, 0, 1]
diamond_y = [0, 1, 0, -1, 0]
ax.plot(diamond_x, diamond_y, 'r-', lw=2, label='‖v‖₁ = 1')
ax.set_title('L1 范数等高线（菱形）\n正则化效果：把部分权重推向精确的零', fontsize=12, fontweight='bold')
ax.set_xlabel('w₁', fontsize=12)
ax.set_ylabel('w₂', fontsize=12)
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=10)

fig.suptitle('L1 vs L2 范数：形状决定正则化行为', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('/tmp/llm-math-foundations/chapter01-linear-algebra/images/l1_l2_norm_contour.png', dpi=150, bbox_inches='tight')
print("✅ 已保存: images/l1_l2_norm_contour.png")
