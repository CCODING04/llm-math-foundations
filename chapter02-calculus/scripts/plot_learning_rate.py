#!/usr/bin/env python3
"""不同学习率的影响"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'Heiti TC']
matplotlib.rcParams['axes.unicode_minus'] = False

# 一维例子：f(w) = w² + sin(3w)
def f(w):
    return w**2 + np.sin(3*w)

def grad_f(w):
    return 2*w + 3*np.cos(3*w)

def gradient_descent_1d(start, lr, n_steps):
    path = [start]
    w = start
    for _ in range(n_steps):
        w = w - lr * grad_f(w)
        path.append(w)
    return np.array(path)

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
w_range = np.linspace(-5, 5, 300)
f_vals = f(w_range)

learning_rates = [
    (0.01, '太小 (η=0.01)', 'green'),
    (0.1, '刚好 (η=0.1)', 'blue'),
    (0.5, '太大 (η=0.5)', 'red'),
]

for ax, (lr, title, color) in zip(axes, learning_rates):
    path = gradient_descent_1d(start=3.0, lr=lr, n_steps=30)
    
    ax.plot(w_range, f_vals, 'k-', linewidth=1.5, alpha=0.5)
    
    # 绘制路径
    f_path = f(path)
    ax.plot(path, f_path, 'o-', color=color, markersize=4, linewidth=1.5)
    ax.plot(path[0], f_path[0], 'rs', markersize=10, label=f'起点 w={path[0]:.1f}')
    ax.plot(path[-1], f_path[-1], 'g*', markersize=12, label=f'终点 w={path[-1]:.2f}')
    
    ax.set_title(title, fontsize=13)
    ax.set_xlabel('w', fontsize=11)
    ax.set_ylabel('f(w)', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-3, 15)

plt.suptitle('学习率 η 对梯度下降的影响', fontsize=15, y=1.02)
plt.tight_layout()
plt.savefig('/tmp/llm-math-foundations/chapter02-calculus/images/learning_rate.png', dpi=150, bbox_inches='tight')
print("✅ Saved: images/learning_rate.png")
