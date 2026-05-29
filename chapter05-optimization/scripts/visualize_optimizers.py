#!/usr/bin/env python3
"""
优化器可视化：从零实现 SGD 和 Adam，对比收敛轨迹
生成图片保存到 ../images/ 目录
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'STHeiti', 'PingFang SC']
plt.rcParams['axes.unicode_minus'] = False

OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'images')
os.makedirs(OUT_DIR, exist_ok=True)


# ============================================================
# 测试函数：Rosenbrock 和 简单二次函数
# ============================================================

def rosenbrock(w):
    """Rosenbrock 函数：经典的非凸优化测试函数"""
    x, y = w[0], w[1]
    return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2

def rosenbrock_grad(w):
    """Rosenbrock 梯度"""
    x, y = w[0], w[1]
    dx = -2 * (1 - x) - 400 * x * (y - x ** 2)
    dy = 200 * (y - x ** 2)
    return np.array([dx, dy])

def quadratic(w):
    """简单二次函数：f(x,y) = 3x^2 + y^2"""
    return 3 * w[0] ** 2 + w[1] ** 2

def quadratic_grad(w):
    return np.array([6 * w[0], 2 * w[1]])


# ============================================================
# 优化器实现
# ============================================================

def sgd(grad_fn, w0, lr=0.01, steps=200):
    """Vanilla SGD"""
    w = w0.copy()
    path = [w.copy()]
    for _ in range(steps):
        g = grad_fn(w)
        w = w - lr * g
        path.append(w.copy())
    return np.array(path)

def sgd_momentum(grad_fn, w0, lr=0.01, momentum=0.9, steps=200):
    """SGD + 动量"""
    w = w0.copy()
    v = np.zeros_like(w)  # 速度
    path = [w.copy()]
    for _ in range(steps):
        g = grad_fn(w)
        v = momentum * v - lr * g
        w = w + v
        path.append(w.copy())
    return np.array(path)

def adam(grad_fn, w0, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8, steps=200):
    """Adam 优化器：完整实现，包含偏差修正"""
    w = w0.copy()
    m = np.zeros_like(w)  # 一阶矩估计
    v = np.zeros_like(w)  # 二阶矩估计
    path = [w.copy()]
    for t in range(1, steps + 1):
        g = grad_fn(w)
        # 更新一阶矩
        m = beta1 * m + (1 - beta1) * g
        # 更新二阶矩
        v = beta2 * v + (1 - beta2) * g ** 2
        # 偏差修正
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        # 参数更新
        w = w - lr * m_hat / (np.sqrt(v_hat) + eps)
        path.append(w.copy())
    return np.array(path)


# ============================================================
# 绘图工具
# ============================================================

def plot_contour_with_path(fn, paths, labels, title, filename,
                           x_range=(-2, 2), y_range=(-1, 3), levels=50):
    """绘制等高线 + 优化路径"""
    fig, ax = plt.subplots(figsize=(10, 7))

    # 等高线
    x = np.linspace(x_range[0], x_range[1], 300)
    y = np.linspace(y_range[0], y_range[1], 300)
    X, Y = np.meshgrid(x, y)
    Z = np.array([fn(np.array([xi, yi])) for xi, yi in zip(X.ravel(), Y.ravel())])
    Z = Z.reshape(X.shape)
    # 对数尺度让等高线更均匀
    ax.contourf(X, Y, np.log1p(Z), levels=levels, cmap='YlOrRd', alpha=0.6)
    ax.contour(X, Y, np.log1p(Z), levels=20, colors='gray', alpha=0.3, linewidths=0.5)

    colors = ['#2196F3', '#4CAF50', '#FF5722']
    for i, (path, label) in enumerate(zip(paths, labels)):
        color = colors[i % len(colors)]
        ax.plot(path[:, 0], path[:, 1], '-o', color=color, markersize=2,
                linewidth=1.5, label=label, alpha=0.8)
        ax.plot(path[0, 0], path[0, 1], 's', color=color, markersize=10)
        ax.plot(path[-1, 0], path[-1, 1], '*', color=color, markersize=15)

    ax.plot(1, 1, 'k*', markersize=20, label='最优点 (1,1)')

    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='upper left')
    plt.tight_layout()
    filepath = os.path.join(OUT_DIR, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'✅ 保存: {filepath}')


def plot_loss_curves(fn, paths, labels, title, filename):
    """绘制损失随迭代步数的变化"""
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#2196F3', '#4CAF50', '#FF5722']
    for i, (path, label) in enumerate(zip(paths, labels)):
        color = colors[i % len(colors)]
        losses = [fn(w) for w in path]
        ax.plot(losses, color=color, linewidth=2, label=label)

    ax.set_xlabel('迭代步数', fontsize=12)
    ax.set_ylabel('损失值 (log scale)', fontsize=12)
    ax.set_yscale('log')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    filepath = os.path.join(OUT_DIR, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'✅ 保存: {filepath}')


# ============================================================
# 实验 1：二次函数上的对比
# ============================================================

print("=" * 50)
print("实验 1：二次函数 f(x,y) = 3x² + y²")
print("=" * 50)

w0_quad = np.array([1.8, 1.5])
steps_quad = 100

path_sgd_q = sgd(quadratic_grad, w0_quad, lr=0.05, steps=steps_quad)
path_mom_q = sgd_momentum(quadratic_grad, w0_quad, lr=0.05, momentum=0.9, steps=steps_quad)
path_adam_q = adam(quadratic_grad, w0_quad, lr=0.1, steps=steps_quad)

plot_contour_with_path(
    quadratic,
    [path_sgd_q, path_mom_q, path_adam_q],
    ['SGD (lr=0.05)', 'SGD+Momentum (lr=0.05)', 'Adam (lr=0.1)'],
    '二次函数上的优化器对比',
    'optimizer_quadratic_path.png',
    x_range=(-2.5, 2.5), y_range=(-2, 2)
)

plot_loss_curves(
    quadratic,
    [path_sgd_q, path_mom_q, path_adam_q],
    ['SGD', 'SGD+Momentum', 'Adam'],
    '二次函数：损失值随迭代步数变化',
    'optimizer_quadratic_loss.png'
)


# ============================================================
# 实验 2：Rosenbrock 函数上的对比
# ============================================================

print("\n" + "=" * 50)
print("实验 2：Rosenbrock 函数（非凸）")
print("=" * 50)

w0_rosen = np.array([-1.0, 2.5])
steps_rosen = 500

path_sgd_r = sgd(rosenbrock_grad, w0_rosen, lr=0.001, steps=steps_rosen)
path_mom_r = sgd_momentum(rosenbrock_grad, w0_rosen, lr=0.001, momentum=0.9, steps=steps_rosen)
path_adam_r = adam(rosenbrock_grad, w0_rosen, lr=0.01, steps=steps_rosen)

plot_contour_with_path(
    rosenbrock,
    [path_sgd_r, path_mom_r, path_adam_r],
    ['SGD (lr=0.001)', 'SGD+Momentum (lr=0.001)', 'Adam (lr=0.01)'],
    'Rosenbrock 函数上的优化器对比',
    'optimizer_rosenbrock_path.png',
    x_range=(-1.5, 2), y_range=(-0.5, 3)
)

plot_loss_curves(
    rosenbrock,
    [path_sgd_r, path_mom_r, path_adam_r],
    ['SGD', 'SGD+Momentum', 'Adam'],
    'Rosenbrock 函数：损失值随迭代步数变化',
    'optimizer_rosenbrock_loss.png'
)


# ============================================================
# 实验 3：不同学习率对 SGD 的影响
# ============================================================

print("\n" + "=" * 50)
print("实验 3：学习率对 SGD 收敛的影响")
print("=" * 50)

lrs = [0.001, 0.01, 0.05, 0.15]
paths_lr = []
labels_lr = []

for lr in lrs:
    path = sgd(quadratic_grad, w0_quad, lr=lr, steps=50)
    paths_lr.append(path)
    labels_lr.append(f'lr={lr}')

fig, ax = plt.subplots(figsize=(10, 7))
x = np.linspace(-2.5, 2.5, 300)
y = np.linspace(-2, 2, 300)
X, Y = np.meshgrid(x, y)
Z = 3 * X ** 2 + Y ** 2
ax.contourf(X, Y, Z, levels=30, cmap='YlOrRd', alpha=0.5)
ax.contour(X, Y, Z, levels=15, colors='gray', alpha=0.3, linewidths=0.5)

colors_lr = ['#9C27B0', '#2196F3', '#4CAF50', '#FF5722']
for i, (path, label) in enumerate(zip(paths_lr, labels_lr)):
    ax.plot(path[:, 0], path[:, 1], '-o', color=colors_lr[i],
            markersize=2, linewidth=1.5, label=label, alpha=0.8)
    ax.plot(path[0, 0], path[0, 1], 's', color=colors_lr[i], markersize=8)
    ax.plot(path[-1, 0], path[-1, 1], '*', color=colors_lr[i], markersize=12)

ax.plot(0, 0, 'k*', markersize=20, label='最优点 (0,0)')
ax.set_xlabel('x', fontsize=12)
ax.set_ylabel('y', fontsize=12)
ax.set_title('不同学习率对 SGD 收敛的影响', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
plt.tight_layout()
filepath = os.path.join(OUT_DIR, 'learning_rate_effect.png')
plt.savefig(filepath, dpi=150, bbox_inches='tight')
plt.close()
print(f'✅ 保存: {filepath}')


# ============================================================
# 实验 4：Adam 内部状态可视化
# ============================================================

print("\n" + "=" * 50)
print("实验 4：Adam 内部状态（一阶矩 m 和二阶矩 v）")
print("=" * 50)

w0_demo = np.array([1.8, 1.5])
w = w0_demo.copy()
m = np.zeros(2)
v = np.zeros(2)
beta1, beta2, eps = 0.9, 0.999, 1e-8
lr = 0.1

history_m = []
history_v = []
history_mhat = []
history_vhat = []
history_grad = []

for t in range(1, 51):
    g = quadratic_grad(w)
    m = beta1 * m + (1 - beta1) * g
    v = beta2 * v + (1 - beta2) * g ** 2
    m_hat = m / (1 - beta1 ** t)
    v_hat = v / (1 - beta2 ** t)
    w = w - lr * m_hat / (np.sqrt(v_hat) + eps)

    history_grad.append(g.copy())
    history_m.append(m.copy())
    history_v.append(v.copy())
    history_mhat.append(m_hat.copy())
    history_vhat.append(v_hat.copy())

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

steps_arr = np.arange(1, 51)

# 梯度
axes[0, 0].plot(steps_arr, [g[0] for g in history_grad], label='g_x', linewidth=2)
axes[0, 0].plot(steps_arr, [g[1] for g in history_grad], label='g_y', linewidth=2)
axes[0, 0].set_title('梯度 g', fontsize=13, fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].set_xlabel('步数')

# 一阶矩 m 和修正后的 m_hat
axes[0, 1].plot(steps_arr, [m[0] for m in history_m], label='m_x', linewidth=2, linestyle='--')
axes[0, 1].plot(steps_arr, [m[0] for m in history_mhat], label='m̂_x (修正)', linewidth=2)
axes[0, 1].set_title('一阶矩 m vs 修正后 m̂', fontsize=13, fontweight='bold')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].set_xlabel('步数')

# 二阶矩 v 和修正后的 v_hat
axes[1, 0].plot(steps_arr, [v[0] for v in history_v], label='v_x', linewidth=2, linestyle='--')
axes[1, 0].plot(steps_arr, [v[0] for v in history_vhat], label='v̂_x (修正)', linewidth=2)
axes[1, 0].set_title('二阶矩 v vs 修正后 v̂', fontsize=13, fontweight='bold')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].set_xlabel('步数')

# 有效步长 lr * m_hat / (sqrt(v_hat) + eps)
effective_lr_x = [lr * history_mhat[i][0] / (np.sqrt(history_vhat[i][0]) + eps) for i in range(50)]
effective_lr_y = [lr * history_mhat[i][1] / (np.sqrt(history_vhat[i][1]) + eps) for i in range(50)]
axes[1, 1].plot(steps_arr, effective_lr_x, label='有效步长_x', linewidth=2)
axes[1, 1].plot(steps_arr, effective_lr_y, label='有效步长_y', linewidth=2)
axes[1, 1].set_title('Adam 有效步长 (自适应!)', fontsize=13, fontweight='bold')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].set_xlabel('步数')

plt.suptitle('Adam 优化器内部状态可视化', fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
filepath = os.path.join(OUT_DIR, 'adam_internals.png')
plt.savefig(filepath, dpi=150, bbox_inches='tight')
plt.close()
print(f'✅ 保存: {filepath}')


# ============================================================
# 实验 5：学习率调度策略可视化
# ============================================================

print("\n" + "=" * 50)
print("实验 5：学习率调度策略")
print("=" * 50)

total_steps = 1000
warmup_steps = 100
peak_lr = 1e-3

def cosine_schedule(step, total=total_steps, warmup=warmup_steps, peak=peak_lr):
    if step < warmup:
        return peak * step / warmup
    progress = (step - warmup) / (total - warmup)
    return peak * 0.5 * (1 + np.cos(np.pi * progress))

def linear_warmup_decay(step, total=total_steps, warmup=warmup_steps, peak=peak_lr):
    if step < warmup:
        return peak * step / warmup
    return peak * (1 - (step - warmup) / (total - warmup))

def constant_schedule(step, peak=peak_lr):
    return peak

def step_decay(step, total=total_steps, warmup=warmup_steps, peak=peak_lr):
    if step < warmup:
        return peak * step / warmup
    progress = (step - warmup) / (total - warmup)
    if progress < 0.33:
        return peak
    elif progress < 0.66:
        return peak * 0.1
    else:
        return peak * 0.01

steps_arr = np.arange(total_steps)
cosine_lrs = [cosine_schedule(s) for s in steps_arr]
linear_lrs = [linear_warmup_decay(s) for s in steps_arr]
constant_lrs = [constant_schedule(s) for s in steps_arr]
step_lrs = [step_decay(s) for s in steps_arr]

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(steps_arr, cosine_lrs, linewidth=2.5, label='余弦退火 (Cosine Annealing)', color='#2196F3')
ax.plot(steps_arr, linear_lrs, linewidth=2, label='线性衰减 (Linear Decay)', color='#4CAF50', linestyle='--')
ax.plot(steps_arr, step_lrs, linewidth=2, label='阶梯衰减 (Step Decay)', color='#FF9800', linestyle='-.')
ax.plot(steps_arr, constant_lrs, linewidth=1.5, label='常数学习率', color='gray', alpha=0.5, linestyle=':')

# 标注 warmup 区间
ax.axvspan(0, warmup_steps, alpha=0.1, color='orange', label='Warmup 阶段')
ax.axvline(x=warmup_steps, color='orange', linestyle='--', alpha=0.5)

ax.set_xlabel('训练步数', fontsize=12)
ax.set_ylabel('学习率', fontsize=12)
ax.set_title('常见学习率调度策略对比', fontsize=14, fontweight='bold')
ax.legend(fontsize=10, loc='upper right')
ax.grid(True, alpha=0.3)
plt.tight_layout()
filepath = os.path.join(OUT_DIR, 'lr_schedules.png')
plt.savefig(filepath, dpi=150, bbox_inches='tight')
plt.close()
print(f'✅ 保存: {filepath}')

print("\n🎉 所有图片生成完毕！")
