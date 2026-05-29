#!/usr/bin/env python3
"""导数的几何意义：函数曲线与切线"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'Heiti TC']
matplotlib.rcParams['axes.unicode_minus'] = False

x = np.linspace(-2, 4, 200)
y = x**3 - 3*x**2 + 2*x + 1  # f(x) = x³ - 3x² + 2x + 1

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# --- 左图：割线到切线 ---
ax = axes[0]
ax.plot(x, y, 'b-', linewidth=2, label=r'$f(x) = x^3 - 3x^2 + 2x + 1$')

x0 = 1.5
f_x0 = x0**3 - 3*x0**2 + 2*x0 + 1
# 导数 f'(x) = 3x² - 6x + 2, f'(1.5) = 6.75 - 9 + 2 = -0.25
f_prime_x0 = 3*x0**2 - 6*x0 + 2

# 切线
tangent = f_prime_x0 * (x - x0) + f_x0
ax.plot(x, tangent, 'r-', linewidth=2, label=f'切线 (斜率 = {f_prime_x0:.2f})')
ax.plot(x0, f_x0, 'ro', markersize=10, zorder=5)

# 割线（Δx = 2）
dx_big = 2.0
x1_big = x0 + dx_big
f_x1_big = x1_big**3 - 3*x1_big**2 + 2*x1_big + 1
secant_slope_big = (f_x1_big - f_x0) / dx_big
secant_big = secant_slope_big * (x - x0) + f_x0
ax.plot(x, secant_big, 'g--', linewidth=1.5, alpha=0.7, label=f'割线 Δx={dx_big} (斜率={secant_slope_big:.2f})')

# 割线（Δx = 0.5）
dx_small = 0.5
x1_small = x0 + dx_small
f_x1_small = x1_small**3 - 3*x1_small**2 + 2*x1_small + 1
secant_slope_small = (f_x1_small - f_x0) / dx_small
secant_small = secant_slope_small * (x - x0) + f_x0
ax.plot(x, secant_small, 'm--', linewidth=1.5, alpha=0.7, label=f'割线 Δx={dx_small} (斜率={secant_slope_small:.2f})')

ax.set_xlim(-1, 4)
ax.set_ylim(-3, 8)
ax.set_xlabel('x', fontsize=12)
ax.set_ylabel('f(x)', fontsize=12)
ax.set_title('从割线到切线：Δx → 0', fontsize=14)
ax.legend(fontsize=9, loc='upper left')
ax.grid(True, alpha=0.3)

# --- 右图：导函数 ---
ax2 = axes[1]
y_prime = 3*x**2 - 6*x + 2
ax2.plot(x, y, 'b-', linewidth=2, label='f(x)')
ax2.plot(x, y_prime, 'r-', linewidth=2, label="f'(x)")
ax2.axhline(y=0, color='k', linewidth=0.5)

# 标记极值点
# f'(x) = 0 → 3x² - 6x + 2 = 0 → x = (6 ± √12)/6
roots = [(6 + np.sqrt(12))/6, (6 - np.sqrt(12))/6]
for r in roots:
    fr = r**3 - 3*r**2 + 2*r + 1
    label = '极小值' if 6*r - 6 > 0 else '极大值'
    ax2.plot(r, fr, 'go', markersize=8, zorder=5)
    ax2.annotate(label, (r, fr), textcoords="offset points", xytext=(10, 10), fontsize=10, color='green')

ax2.set_xlabel('x', fontsize=12)
ax2.set_ylabel('y', fontsize=12)
ax2.set_title('函数与导函数', fontsize=14)
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/tmp/llm-math-foundations/chapter02-calculus/images/derivative_tangent.png', dpi=150, bbox_inches='tight')
print("✅ Saved: images/derivative_tangent.png")
