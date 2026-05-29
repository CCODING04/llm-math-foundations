"""
二维向量可视化
展示向量的几何意义：方向和长度
"""
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['PingFang SC', 'Heiti SC', 'STHeiti', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['text.usetex'] = False  # 使用 mathtext 而非 full LaTeX
import numpy as np

fig, ax = plt.subplots(1, 1, figsize=(8, 6))

# 画几个向量
vectors = {
    r'$v_1$ = [2, 3]': np.array([2, 3]),
    r'$v_2$ = [-1, 2]': np.array([-1, 2]),
    r'$v_3$ = [3, -1]': np.array([3, -1]),
}

colors = ['#2196F3', '#FF5722', '#4CAF50']

for (label, v), color in zip(vectors.items(), colors):
    ax.annotate('', xy=v, xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color=color, lw=2.5))
    ax.text(v[0] + 0.15, v[1] + 0.15, label, fontsize=12, color=color, fontweight='bold')
    # 标注向量长度
    norm = np.linalg.norm(v)
    mid = v / 2
    ax.text(mid[0] - 0.3, mid[1] - 0.3, f'$||v||$={norm:.2f}', fontsize=9, color=color,
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor=color, alpha=0.8))

# 坐标轴
ax.axhline(y=0, color='gray', linewidth=0.5)
ax.axvline(x=0, color='gray', linewidth=0.5)
ax.set_xlim(-2.5, 4.5)
ax.set_ylim(-2, 4.5)
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)
ax.set_xlabel('x', fontsize=12)
ax.set_ylabel('y', fontsize=12)
ax.set_title('二维向量可视化：每个向量都有方向和长度', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('/tmp/llm-math-foundations/chapter01-linear-algebra/images/vectors_2d.png', dpi=150, bbox_inches='tight')
print("✅ 已保存: images/vectors_2d.png")
