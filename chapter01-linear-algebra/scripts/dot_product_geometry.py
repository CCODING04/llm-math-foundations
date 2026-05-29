"""
点积与向量夹角的关系
展示点积如何反映向量间的相似度
"""
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['PingFang SC', 'Heiti SC', 'STHeiti', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False
import numpy as np

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

v = np.array([1, 0])
targets = [
    (np.array([0.87, 0.5]), 'θ≈30° 同向', '#4CAF50'),
    (np.array([0, 1]), 'θ=90° 垂直', '#FF9800'),
    (np.array([-0.87, 0.5]), 'θ≈150° 反向', '#F44336'),
]

for ax, (t, label, color) in zip(axes, targets):
    # 画向量 v
    ax.annotate('', xy=v, xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color='#2196F3', lw=2.5))
    ax.text(v[0] + 0.05, v[1] - 0.15, 'a', fontsize=13, color='#2196F3', fontweight='bold')

    # 画向量 t
    ax.annotate('', xy=t, xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color=color, lw=2.5))
    ax.text(t[0] + 0.05, t[1] + 0.05, 'b', fontsize=13, color=color, fontweight='bold')

    # 画夹角弧线
    theta_v = np.arctan2(v[1], v[0])
    theta_t = np.arctan2(t[1], t[0])
    thetas = np.linspace(theta_v, theta_t, 30)
    arc_r = 0.3
    ax.plot(arc_r * np.cos(thetas), arc_r * np.sin(thetas), color='gray', lw=1.5)

    # 计算点积
    dot = np.dot(v, t)
    cos_theta = dot / (np.linalg.norm(v) * np.linalg.norm(t))

    ax.set_title(f'{label}\na·b = {dot:.2f}, cos(θ) = {cos_theta:.2f}',
                 fontsize=12, fontweight='bold', color=color)

    ax.set_xlim(-1.3, 1.5)
    ax.set_ylim(-0.5, 1.3)
    ax.set_aspect('equal')
    ax.axhline(y=0, color='gray', linewidth=0.5)
    ax.axvline(x=0, color='gray', linewidth=0.5)
    ax.grid(True, alpha=0.3)

fig.suptitle('点积与夹角：方向越一致，点积越大', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('/tmp/llm-math-foundations/chapter01-linear-algebra/images/dot_product_geometry.png', dpi=150, bbox_inches='tight')
print("✅ 已保存: images/dot_product_geometry.png")
