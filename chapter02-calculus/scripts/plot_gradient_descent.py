#!/usr/bin/env python3
"""梯度下降过程可视化：等高线 + 下降路径"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'Heiti TC']
matplotlib.rcParams['axes.unicode_minus'] = False

# 目标函数：f(w1, w2) = w1² + 2w2²（碗形）
def f(w1, w2):
    return w1**2 + 2*w2**2

def grad_f(w1, w2):
    return np.array([2*w1, 4*w2])

# 梯度下降
def gradient_descent(start, lr, n_steps):
    path = [start.copy()]
    w = start.copy()
    for _ in range(n_steps):
        g = grad_f(w[0], w[1])
        w = w - lr * g
        path.append(w.copy())
    return np.array(path)

# 运行梯度下降
start = np.array([3.0, 2.5])
path = gradient_descent(start, lr=0.1, n_steps=30)

# 绘图
fig, ax = plt.subplots(figsize=(8, 7))

# 等高线
w1_range = np.linspace(-4, 4, 200)
w2_range = np.linspace(-4, 4, 200)
W1, W2 = np.meshgrid(w1_range, w2_range)
Z = f(W1, W2)

contour = ax.contour(W1, W2, Z, levels=20, cmap='viridis', alpha=0.7)
ax.clabel(contour, inline=True, fontsize=8, fmt='%.1f')

# 下降路径
ax.plot(path[:, 0], path[:, 1], 'r-o', markersize=5, linewidth=1.5, label='梯度下降路径', zorder=5)
ax.plot(path[0, 0], path[0, 1], 'rs', markersize=12, label=f'起点 ({path[0,0]:.1f}, {path[0,1]:.1f})', zorder=6)
ax.plot(path[-1, 0], path[-1, 1], 'g*', markersize=15, label=f'终点 ({path[-1,0]:.2f}, {path[-1,1]:.2f})', zorder=6)

# 在几个点画梯度箭头（反方向 = 下降方向）
for i in range(0, len(path)-1, 5):
    g = grad_f(path[i, 0], path[i, 1])
    g_norm = g / (np.linalg.norm(g) + 1e-8) * 0.5
    ax.annotate('', xy=(path[i, 0] - g_norm[0], path[i, 1] - g_norm[1]),
                xytext=(path[i, 0], path[i, 1]),
                arrowprops=dict(arrowstyle='->', color='blue', lw=2))

ax.set_xlabel('$w_1$', fontsize=13)
ax.set_ylabel('$w_2$', fontsize=13)
ax.set_title(r'梯度下降：$f(w_1, w_2) = w_1^2 + 2w_2^2$, $\eta = 0.1$', fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_aspect('equal')

plt.tight_layout()
plt.savefig('/tmp/llm-math-foundations/chapter02-calculus/images/gradient_descent.png', dpi=150, bbox_inches='tight')
print("✅ Saved: images/gradient_descent.png")
