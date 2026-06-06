#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
阶段项目 1：从零实现线性回归
==============================

完整可运行脚本，包含 7 个步骤：
1. 用 NumPy 构造数据（y = 3x + 2 + noise）
2. 手写前向传播：y_hat = X @ w + b
3. 手写损失函数：MSE = mean((y - y_hat)²)
4. 手写梯度：∂L/∂w, ∂L/∂b
5. 梯度下降循环：w -= lr * grad
6. 画出 loss 曲线和拟合结果
7. 验证：MSE 最小解 = 正规方程解 = MLE 解

依赖：numpy, matplotlib
运行：python linear_regression.py
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# 配置中文字体（如果可用的话）
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'STSong', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 确保输出目录存在
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
IMAGE_DIR = os.path.join(PROJECT_DIR, 'images')
os.makedirs(IMAGE_DIR, exist_ok=True)

# 设置随机种子，保证可复现
np.random.seed(42)

print("=" * 60)
print("  🚀 阶段项目 1：从零实现线性回归")
print("=" * 60)
print()

# ============================================================
# 步骤 1：用 NumPy 构造数据
# ============================================================
# 真实关系：y = 3x + 2 + ε，其中 ε ~ N(0, σ²)
# 这模拟了"信号 + 噪声"的真实数据场景

print("=" * 60)
print("  步骤 1：构造数据")
print("=" * 60)

n_samples = 100       # 样本数量
true_w = 3.0          # 真实斜率
true_b = 2.0          # 真实截距
noise_std = 1.0       # 噪声标准差

# 生成 x：在 [0, 10] 之间均匀采样
X = np.random.uniform(0, 10, size=n_samples)

# 生成高斯噪声：ε ~ N(0, σ²)
noise = np.random.normal(0, noise_std, size=n_samples)

# 生成 y = 3x + 2 + noise
y = true_w * X + true_b + noise

print(f"  数据量: {n_samples}")
print(f"  X 范围: [{X.min():.2f}, {X.max():.2f}]")
print(f"  y 范围: [{y.min():.2f}, {y.max():.2f}]")
print(f"  真实参数: w={true_w}, b={true_b}")
print(f"  噪声标准差: {noise_std}")
print()

# ============================================================
# 步骤 2：手写前向传播
# ============================================================
# 前向传播：给定参数 (w, b)，计算预测值 y_hat
# y_hat = X * w + b
# 这就是 Ch1 学的矩阵-向量乘法 + 偏置加法

print("=" * 60)
print("  步骤 2：前向传播")
print("=" * 60)

def forward(X, w, b):
    """前向传播：y_hat = X * w + b"""
    return X * w + b

# 初始化参数（从 0 开始）
w = 0.0
b = 0.0

y_hat_init = forward(X, w, b)
print(f"  初始参数: w={w:.4f}, b={b:.4f}")
print(f"  初始预测范围: [{y_hat_init.min():.2f}, {y_hat_init.max():.2f}]")
print()

# ============================================================
# 步骤 3：手写损失函数（MSE）
# ============================================================
# MSE = (1/n) * Σ(y_i - y_hat_i)²
# 衡量预测值和真实值的平均偏差

print("=" * 60)
print("  步骤 3：损失函数（MSE）")
print("=" * 60)

def mse_loss(y_true, y_pred):
    """计算均方误差损失"""
    return np.mean((y_true - y_pred) ** 2)

loss_init = mse_loss(y, y_hat_init)
print(f"  初始 MSE 损失: {loss_init:.4f}")
print()

# ============================================================
# 步骤 4：手写梯度
# ============================================================
# 用链式法则推导：
#   ∂L/∂w = -(2/n) * Σ(y_i - y_hat_i) * x_i
#   ∂L/∂b = -(2/n) * Σ(y_i - y_hat_i)
# 
# 这就是 Ch2 学的偏导数 + 链式法则的实战！

print("=" * 60)
print("  步骤 4：梯度计算")
print("=" * 60)

def compute_gradients(X, y, w, b):
    """
    手写梯度计算
    
    推导过程：
    L = (1/n) Σ(y_i - (wx_i + b))²
    
    ∂L/∂w = (1/n) Σ 2(y_i - ŷ_i) * (-x_i) = -(2/n) Σ(y_i - ŷ_i) * x_i
    ∂L/∂b = (1/n) Σ 2(y_i - ŷ_i) * (-1)   = -(2/n) Σ(y_i - ŷ_i)
    """
    n = len(y)
    y_hat = forward(X, w, b)
    error = y - y_hat  # 残差
    
    # 对 w 的梯度
    dw = -2.0 / n * np.sum(error * X)
    
    # 对 b 的梯度
    db = -2.0 / n * np.sum(error)
    
    return dw, db

dw, db = compute_gradients(X, y, w, b)
print(f"  初始梯度: dw={dw:.4f}, db={db:.4f}")
print()

# ============================================================
# 步骤 5：梯度下降循环
# ============================================================
# 参数更新规则（Ch2 §2.5）：
#   w ← w - η * ∂L/∂w
#   b ← b - η * ∂L/∂b
# 
# 学习率 η 控制每一步迈多大

print("=" * 60)
print("  步骤 5：梯度下降训练")
print("=" * 60)

# 训练超参数
learning_rate = 0.01     # 学习率
n_epochs = 100           # 训练轮数

# 重新初始化参数
w = 0.0
b = 0.0

# 记录训练过程
loss_history = []
w_history = []
b_history = []

print(f"  训练配置: lr={learning_rate}, epochs={n_epochs}")
print(f"  {'Epoch':>6s}  {'Loss':>10s}  {'w':>10s}  {'b':>10s}")
print(f"  {'-'*6}  {'-'*10}  {'-'*10}  {'-'*10}")

for epoch in range(n_epochs):
    # 前向传播
    y_hat = forward(X, w, b)
    
    # 计算损失
    loss = mse_loss(y, y_hat)
    loss_history.append(loss)
    
    # 计算梯度
    dw, db = compute_gradients(X, y, w, b)
    
    # 更新参数（梯度下降的核心！）
    w -= learning_rate * dw
    b -= learning_rate * db
    
    # 记录参数变化
    w_history.append(w)
    b_history.append(b)
    
    # 定期打印训练进度
    if epoch % 20 == 0 or epoch == n_epochs - 1:
        print(f"  {epoch:6d}  {loss:10.4f}  {w:10.4f}  {b:10.4f}")

print()
print(f"  ✅ 训练完成！")
print(f"  最终参数: w={w:.6f} (真实值={true_w})")
print(f"            b={b:.6f} (真实值={true_b})")
print(f"  最终损失: {loss_history[-1]:.6f}")
print()

# ============================================================
# 步骤 6：画出 loss 曲线和拟合结果
# ============================================================

print("=" * 60)
print("  步骤 6：可视化")
print("=" * 60)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# --- 图1：Loss 曲线 ---
axes[0].plot(loss_history, linewidth=2, color='#2196F3')
axes[0].set_xlabel('Epoch', fontsize=12)
axes[0].set_ylabel('MSE Loss', fontsize=12)
axes[0].set_title('Training Loss Curve', fontsize=14)
axes[0].grid(True, alpha=0.3)

# --- 图2：参数收敛过程 ---
axes[1].plot(w_history, linewidth=2, label=f'w (learned)', color='#FF5722')
axes[1].axhline(y=true_w, color='red', linestyle='--', linewidth=1.5,
                label=f'w (true={true_w})')
axes[1].plot(b_history, linewidth=2, label=f'b (learned)', color='#4CAF50')
axes[1].axhline(y=true_b, color='green', linestyle='--', linewidth=1.5,
                label=f'b (true={true_b})')
axes[1].set_xlabel('Epoch', fontsize=12)
axes[1].set_ylabel('Parameter Value', fontsize=12)
axes[1].set_title('Parameter Convergence', fontsize=14)
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)

# --- 图3：拟合结果 ---
axes[2].scatter(X, y, alpha=0.5, s=20, label='Data Points', color='#2196F3')
x_line = np.linspace(0, 10, 100)
axes[2].plot(x_line, w * x_line + b, linewidth=2.5, color='#FF5722',
             label=f'Fitted: y={w:.2f}x+{b:.2f}')
axes[2].plot(x_line, true_w * x_line + true_b, linewidth=2, linestyle='--',
             color='#4CAF50', label=f'True: y={true_w}x+{true_b}')
axes[2].set_xlabel('x', fontsize=12)
axes[2].set_ylabel('y', fontsize=12)
axes[2].set_title('Linear Regression Fit', fontsize=14)
axes[2].legend(fontsize=10)
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
save_path = os.path.join(IMAGE_DIR, 'training_result.png')
plt.savefig(save_path, dpi=150, bbox_inches='tight')
print(f"  ✅ 图片已保存: {save_path}")
plt.show()
print()

# ============================================================
# 步骤 7：验证三解合一
# ============================================================
# 梯度下降的解 ≈ 正规方程的解 = MLE 的解

print("=" * 60)
print("  步骤 7：验证三解合一")
print("=" * 60)

# --- 7a. 正规方程（解析解）---
# w_hat = (X^T X)^{-1} X^T y
# 把 b 吸收到 w 里：构造增广矩阵 [X | 1]

X_aug = np.column_stack([X, np.ones(n_samples)])  # shape: (100, 2)

# 正规方程
XtX = X_aug.T @ X_aug
Xty = X_aug.T @ y
w_normal = np.linalg.inv(XtX) @ Xty

print(f"  --- 正规方程（解析解）---")
print(f"  w = {w_normal[0]:.6f}")
print(f"  b = {w_normal[1]:.6f}")
print()

# --- 7b. 梯度下降解 ---
print(f"  --- 梯度下降解 ---")
print(f"  w = {w:.6f}")
print(f"  b = {b:.6f}")
print()

# --- 7c. MLE 理论解释 ---
# 假设 y_i = wx_i + b + ε_i, ε_i ~ N(0, σ²)
# 则 y_i ~ N(wx_i + b, σ²)
# 似然函数 L(w,b) = ∏ (1/√(2πσ²)) exp(-(y_i - wx_i - b)²/(2σ²))
# 取负对数：-log L = n/2 * log(2πσ²) + (1/2σ²) Σ(y_i - ŷ_i)²
# 最小化 -log L 等价于最小化 Σ(y_i - ŷ_i)²，即 MSE！
# 所以：MSE 最小化 = MLE（在高斯噪声假设下）

print(f"  --- MLE 视角 ---")
print(f"  在高斯噪声假设下：")
print(f"  最小化 MSE = 最小化 -log L（负对数似然）")
print(f"  因此 MSE 最优解 = MLE 解")
print()

# --- 三解对比 ---
print(f"  {'='*50}")
print(f"  三解对比:")
print(f"  {'='*50}")
print(f"  {'方法':<12s}  {'w':>12s}  {'b':>12s}  {'MSE':>12s}")
print(f"  {'-'*12}  {'-'*12}  {'-'*12}  {'-'*12}")
print(f"  {'真实值':<12s}  {true_w:12.6f}  {true_b:12.6f}  {'—':>12s}")

mse_gd = mse_loss(y, forward(X, w, b))
mse_ne = mse_loss(y, forward(X, w_normal[0], w_normal[1]))

print(f"  {'梯度下降':<12s}  {w:12.6f}  {b:12.6f}  {mse_gd:12.6f}")
print(f"  {'正规方程':<12s}  {w_normal[0]:12.6f}  {w_normal[1]:12.6f}  {mse_ne:12.6f}")
print()

# 计算差异
delta_w = abs(w - w_normal[0])
delta_b = abs(b - w_normal[1])
delta_mse = abs(mse_gd - mse_ne)

print(f"  梯度下降 vs 正规方程差异:")
print(f"    Δw   = {delta_w:.6f}")
print(f"    Δb   = {delta_b:.6f}")
print(f"    ΔMSE = {delta_mse:.6f}")
print()

if delta_w < 0.1 and delta_b < 0.1:
    print("  ✅ 三解合一验证成功！")
    print("  ✅ MSE 最小解 ≈ 正规方程解 = MLE 解（在高斯噪声假设下）")
else:
    print("  ⚠️  差异较大，可以尝试增加训练轮数或调整学习率")

print()
print("=" * 60)
print("  🎉 项目完成！你从零实现了线性回归的全部流程")
print("=" * 60)
