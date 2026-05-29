#!/usr/bin/env python3
"""定积分 = 曲线下面积"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'Heiti TC']
matplotlib.rcParams['axes.unicode_minus'] = False

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

x = np.linspace(0, np.pi, 300)
y = np.sin(x)

# 精确积分值
exact = -np.cos(np.pi) + np.cos(0)  # = 2.0

# 不同数量的矩形逼近
n_rects = [5, 20, 100]
titles = ['5 个矩形（粗糙）', '20 个矩形（较好）', '100 个矩形（精确）']

for ax, n, title in zip(axes, n_rects, titles):
    # 画函数曲线
    ax.plot(x, y, 'b-', linewidth=2, label=r'$\sin(x)$')
    
    # 画矩形
    dx = np.pi / n
    x_rects = np.linspace(0, np.pi, n, endpoint=False)
    y_rects = np.sin(x_rects + dx/2)  # 中点黎曼和
    
    ax.bar(x_rects + dx/2, y_rects, width=dx, alpha=0.4, color='orange', 
           edgecolor='darkorange', linewidth=0.5)
    
    # 数值计算
    numerical = np.sum(y_rects * dx)
    
    ax.set_title(f'{title}\n面积 ≈ {numerical:.4f} (精确值 = {exact:.4f})', fontsize=12)
    ax.set_xlabel('x', fontsize=11)
    ax.set_ylabel('y', fontsize=11)
    ax.set_xlim(0, np.pi)
    ax.set_ylim(0, 1.15)
    ax.grid(True, alpha=0.3)

plt.suptitle(r'定积分 $\int_0^{\pi} \sin(x)\,dx$ = 曲线下面积', fontsize=15, y=1.03)
plt.tight_layout()
plt.savefig('/tmp/llm-math-foundations/chapter02-calculus/images/integral_area.png', dpi=150, bbox_inches='tight')
print("✅ Saved: images/integral_area.png")
