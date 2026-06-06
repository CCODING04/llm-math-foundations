#!/usr/bin/env python3
"""
阶段项目 2：从零实现两层神经网络
==================================

用纯 NumPy 手写一个两层神经网络，对 make_moons 数据集做二分类。

网络结构：
    h = ReLU(X @ W1 + b1)
    y = softmax(h @ W2 + b2)

损失函数：交叉熵（Cross-Entropy Loss）
优化器：SGD / Adam（手写实现）

运行方式：
    python two_layer_nn.py

依赖：
    pip install numpy matplotlib scikit-learn
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

# ============================================================
# 1. 数据准备：make_moons 二分类数据集
# ============================================================

def prepare_data(n_samples=1000, noise=0.2, random_state=42):
    """生成 make_moons 数据集并划分训练/测试集。"""
    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
    # X: (N, 2), y: (N,) 取值 0 或 1
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )
    return X_train, X_test, y_train, y_test


# ============================================================
# 2. 激活函数 & 辅助函数
# ============================================================

def relu(z):
    """ReLU 激活函数：max(0, z)"""
    return np.maximum(0, z)


def relu_derivative(z):
    """ReLU 的导数：z > 0 时为 1，否则为 0。"""
    return (z > 0).astype(float)


def softmax(z):
    """
    Softmax 函数：将 logits 转换为概率分布。
    
    公式：softmax(z)_i = exp(z_i) / sum(exp(z_j))
    
    数值稳定技巧：先减去最大值，防止 exp 溢出。
    """
    # 数值稳定：减去每行最大值
    z_shifted = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z_shifted)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def one_hot(y, num_classes):
    """将标签转为 one-hot 编码。"""
    N = y.shape[0]
    one_hot_labels = np.zeros((N, num_classes))
    one_hot_labels[np.arange(N), y] = 1.0
    return one_hot_labels


# ============================================================
# 3. 前向传播
# ============================================================

def forward(X, W1, b1, W2, b2):
    """
    两层神经网络的前向传播：
    
        h = ReLU(X @ W1 + b1)    # 隐藏层
        y_hat = softmax(h @ W2 + b2)  # 输出层
    
    参数：
        X:  (N, D)   输入特征
        W1: (D, H)   第一层权重
        b1: (H,)     第一层偏置
        W2: (H, C)   第二层权重
        b2: (C,)     第二层偏置
    
    返回：
        cache: 字典，保存中间结果供反向传播使用
    """
    # 第一层：线性变换 + ReLU
    z1 = X @ W1 + b1          # (N, H)
    h = relu(z1)               # (N, H)
    
    # 第二层：线性变换 + Softmax
    z2 = h @ W2 + b2           # (N, C)
    y_hat = softmax(z2)        # (N, C)
    
    cache = {
        'X': X,
        'z1': z1,
        'h': h,
        'z2': z2,
        'y_hat': y_hat,
    }
    return cache


# ============================================================
# 4. 交叉熵损失（Cross-Entropy Loss）
# ============================================================

def cross_entropy_loss(y_hat, y_onehot):
    """
    交叉熵损失函数：
    
        L = - (1/N) * sum(y_true * log(y_hat))
    
    对应 Ch6 的核心概念：衡量预测分布 Q 与真实分布 P 之间的差异。
    
    参数：
        y_hat:     (N, C) 模型输出的概率分布（softmax 之后）
        y_onehot:  (N, C) 真实标签的 one-hot 编码
    
    返回：
        loss: 标量，平均交叉熵损失
    """
    N = y_hat.shape[0]
    # 裁剪防止 log(0)
    y_hat_clipped = np.clip(y_hat, 1e-12, 1 - 1e-12)
    loss = -np.sum(y_onehot * np.log(y_hat_clipped)) / N
    return loss


# ============================================================
# 5. 反向传播（Backpropagation）
# ============================================================

def backward(cache, y_onehot, W2):
    """
    反向传播：用矩阵求导（Ch4）计算所有参数的梯度。
    
    推导过程：
    
    设 L = CrossEntropy(softmax(z2), y)
    
    1) 输出层梯度（softmax + cross-entropy 的简化）：
       ∂L/∂z2 = y_hat - y      （shape: (N, C)）
       
    2) 第二层参数梯度：
       ∂L/∂W2 = h^T @ (∂L/∂z2)  （shape: (H, C)）
       ∂L/∂b2 = sum(∂L/∂z2, axis=0)  （shape: (C,)）
       
    3) 传播到隐藏层：
       ∂L/∂h = (∂L/∂z2) @ W2^T   （shape: (N, H)）
       ∂L/∂z1 = ∂L/∂h * ReLU'(z1) （shape: (N, H)）
       
    4) 第一层参数梯度：
       ∂L/∂W1 = X^T @ (∂L/∂z1)   （shape: (D, H)）
       ∂L/∂b1 = sum(∂L/∂z1, axis=0)  （shape: (H,)）
    
    参数：
        cache:    前向传播的缓存
        y_onehot: (N, C) one-hot 标签
        W2:       (H, C) 第二层权重（用于计算 ∂L/∂h）
    
    返回：
        grads: 字典，包含所有参数的梯度
    """
    X = cache['X']
    z1 = cache['z1']
    h = cache['h']
    y_hat = cache['y_hat']
    N = X.shape[0]
    
    # 步骤 1：输出层梯度
    # softmax + cross-entropy 组合的导数 = y_hat - y
    dz2 = y_hat - y_onehot              # (N, C)
    
    # 步骤 2：第二层参数梯度
    dW2 = h.T @ dz2 / N                 # (H, C)
    db2 = np.sum(dz2, axis=0) / N       # (C,)
    
    # 步骤 3：传播到隐藏层
    dh = dz2 @ W2.T                     # (N, H)
    dz1 = dh * relu_derivative(z1)      # (N, H)
    
    # 步骤 4：第一层参数梯度
    dW1 = X.T @ dz1 / N                 # (D, H)
    db1 = np.sum(dz1, axis=0) / N       # (H,)
    
    grads = {
        'dW1': dW1,
        'db1': db1,
        'dW2': dW2,
        'db2': db2,
    }
    return grads


# ============================================================
# 6. 参数初始化（He 初始化）
# ============================================================

def init_params(D, H, C, random_state=42):
    """
    He 初始化：适用于 ReLU 激活函数。
    
    W ~ N(0, sqrt(2/fan_in))
    
    这样初始化的好处：让各层的方差保持稳定，避免梯度消失/爆炸。
    """
    rng = np.random.RandomState(random_state)
    W1 = rng.randn(D, H) * np.sqrt(2.0 / D)
    b1 = np.zeros(H)
    W2 = rng.randn(H, C) * np.sqrt(2.0 / H)
    b2 = np.zeros(C)
    return W1, b1, W2, b2


# ============================================================
# 7. 手写优化器：SGD 和 Adam
# ============================================================

class SGD:
    """
    随机梯度下降优化器（带 Momentum 可选）。
    
    参数更新规则：
        v = momentum * v - lr * grad
        param = param + v
    """
    
    def __init__(self, lr=0.01, momentum=0.0):
        self.lr = lr
        self.momentum = momentum
        self.velocities = {}
    
    def step(self, params, grads):
        """执行一步参数更新。"""
        for key in params:
            if key not in self.velocities:
                self.velocities[key] = np.zeros_like(params[key])
            
            v = self.momentum * self.velocities[key] - self.lr * grads[key]
            self.velocities[key] = v
            params[key] = params[key] + v


class Adam:
    """
    Adam 优化器（手写实现）。
    
    Adam = RMSprop + Momentum 的结合体。
    
    更新规则（对应 Ch5 讲解的 Adam）：
        m = β1 * m + (1 - β1) * grad        # 一阶矩估计（动量）
        v = β2 * v + (1 - β2) * grad^2      # 二阶矩估计（自适应学习率）
        m_hat = m / (1 - β1^t)              # 偏差修正
        v_hat = v / (1 - β2^t)              # 偏差修正
        param = param - lr * m_hat / (sqrt(v_hat) + ε)
    
    参数：
        lr:     学习率（默认 0.001）
        beta1:  一阶矩衰减率（默认 0.9）
        beta2:  二阶矩衰减率（默认 0.999）
        eps:    数值稳定项（默认 1e-8）
    """
    
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = {}   # 一阶矩（动量）
        self.v = {}   # 二阶矩（自适应学习率）
        self.t = 0    # 时间步
    
    def step(self, params, grads):
        """执行一步 Adam 参数更新。"""
        self.t += 1
        for key in params:
            if key not in self.m:
                self.m[key] = np.zeros_like(params[key])
                self.v[key] = np.zeros_like(params[key])
            
            g = grads[key]
            
            # 更新一阶矩和二阶矩
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * g
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * g ** 2
            
            # 偏差修正
            m_hat = self.m[key] / (1 - self.beta1 ** self.t)
            v_hat = self.v[key] / (1 - self.beta2 ** self.t)
            
            # 参数更新
            params[key] = params[key] - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


# ============================================================
# 8. 训练循环
# ============================================================

def train(X_train, y_train, X_test, y_test, 
          hidden_size=64, num_classes=2,
          optimizer_type='adam', lr=0.01, 
          epochs=500, batch_size=64,
          random_state=42, verbose=True):
    """
    训练两层神经网络。
    
    参数：
        X_train, y_train: 训练数据
        X_test, y_test:   测试数据
        hidden_size:      隐藏层神经元数
        num_classes:      输出类别数
        optimizer_type:   'sgd' 或 'adam'
        lr:               学习率
        epochs:           训练轮数
        batch_size:       批大小
        random_state:     随机种子
        verbose:          是否打印训练日志
    
    返回：
        history: 训练历史（loss 和 accuracy）
        params:  训练后的参数
    """
    N, D = X_train.shape
    C = num_classes
    
    # 参数初始化
    W1, b1, W2, b2 = init_params(D, hidden_size, C, random_state)
    params = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}
    
    # 创建优化器
    if optimizer_type == 'sgd':
        optimizer = SGD(lr=lr, momentum=0.9)
    elif optimizer_type == 'adam':
        optimizer = Adam(lr=lr)
    else:
        raise ValueError(f"未知优化器: {optimizer_type}")
    
    # 记录训练过程
    history = {
        'train_loss': [],
        'test_loss': [],
        'train_acc': [],
        'test_acc': [],
    }
    
    # 训练标签 one-hot
    y_train_oh = one_hot(y_train, C)
    y_test_oh = one_hot(y_test, C)
    
    rng = np.random.RandomState(random_state)
    
    for epoch in range(1, epochs + 1):
        # --- Mini-batch 训练 ---
        indices = rng.permutation(N)
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            batch_idx = indices[start:end]
            
            X_batch = X_train[batch_idx]
            y_batch_oh = y_train_oh[batch_idx]
            
            # 前向传播
            cache = forward(X_batch, params['W1'], params['b1'], 
                           params['W2'], params['b2'])
            
            # 反向传播
            grads = backward(cache, y_batch_oh, params['W2'])
            
            # 参数更新
            optimizer.step(params, grads)
        
        # --- 记录每个 epoch 的指标 ---
        # 训练集
        train_cache = forward(X_train, params['W1'], params['b1'],
                              params['W2'], params['b2'])
        train_loss = cross_entropy_loss(train_cache['y_hat'], y_train_oh)
        train_acc = np.mean(np.argmax(train_cache['y_hat'], axis=1) == y_train)
        
        # 测试集
        test_cache = forward(X_test, params['W1'], params['b1'],
                             params['W2'], params['b2'])
        test_loss = cross_entropy_loss(test_cache['y_hat'], y_test_oh)
        test_acc = np.mean(np.argmax(test_cache['y_hat'], axis=1) == y_test)
        
        history['train_loss'].append(train_loss)
        history['test_loss'].append(test_loss)
        history['train_acc'].append(train_acc)
        history['test_acc'].append(test_acc)
        
        if verbose and (epoch % 50 == 0 or epoch == 1):
            print(f"[{optimizer_type.upper():>4s}] Epoch {epoch:4d}/{epochs} | "
                  f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                  f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")
    
    return history, params


# ============================================================
# 9. 可视化
# ============================================================

def plot_results(sgd_history, adam_history):
    """绘制 SGD 和 Adam 的训练对比图。"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('两层神经网络：SGD vs Adam 训练对比', fontsize=16, fontweight='bold')
    
    epochs_sgd = range(1, len(sgd_history['train_loss']) + 1)
    epochs_adam = range(1, len(adam_history['train_loss']) + 1)
    
    # --- 1. 训练 Loss 曲线 ---
    ax = axes[0, 0]
    ax.plot(epochs_sgd, sgd_history['train_loss'], label='SGD', color='#e74c3c', alpha=0.8)
    ax.plot(epochs_adam, adam_history['train_loss'], label='Adam', color='#3498db', alpha=0.8)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Cross-Entropy Loss')
    ax.set_title('训练集 Loss 曲线')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # --- 2. 测试 Loss 曲线 ---
    ax = axes[0, 1]
    ax.plot(epochs_sgd, sgd_history['test_loss'], label='SGD', color='#e74c3c', alpha=0.8)
    ax.plot(epochs_adam, adam_history['test_loss'], label='Adam', color='#3498db', alpha=0.8)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Cross-Entropy Loss')
    ax.set_title('测试集 Loss 曲线')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # --- 3. 训练准确率曲线 ---
    ax = axes[1, 0]
    ax.plot(epochs_sgd, sgd_history['train_acc'], label='SGD', color='#e74c3c', alpha=0.8)
    ax.plot(epochs_adam, adam_history['train_acc'], label='Adam', color='#3498db', alpha=0.8)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title('训练集准确率曲线')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.4, 1.05)
    
    # --- 4. 测试准确率曲线 ---
    ax = axes[1, 1]
    ax.plot(epochs_sgd, sgd_history['test_acc'], label='SGD', color='#e74c3c', alpha=0.8)
    ax.plot(epochs_adam, adam_history['test_acc'], label='Adam', color='#3498db', alpha=0.8)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title('测试集准确率曲线')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.4, 1.05)
    
    plt.tight_layout()
    plt.savefig('training_comparison.png', dpi=150, bbox_inches='tight')
    print("\n✅ 训练对比图已保存: training_comparison.png")
    plt.show()


def plot_decision_boundary(X, y, params, title="决策边界"):
    """绘制分类决策边界。"""
    
    h = 0.02  # 网格步长
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # 对网格中每个点做预测
    grid = np.c_[xx.ravel(), yy.ravel()]
    cache = forward(grid, params['W1'], params['b1'], params['W2'], params['b2'])
    Z = np.argmax(cache['y_hat'], axis=1).reshape(xx.shape)
    
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdYlBu)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu, edgecolors='black', s=30)
    plt.title(title, fontsize=14)
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.tight_layout()
    plt.savefig(f'decision_boundary_{title.replace(" ", "_")}.png', dpi=150, bbox_inches='tight')
    print(f"✅ 决策边界图已保存: decision_boundary_{title.replace(' ', '_')}.png")
    plt.show()


# ============================================================
# 10. 主函数
# ============================================================

def main():
    print("=" * 60)
    print("阶段项目 2：从零实现两层神经网络")
    print("=" * 60)
    print()
    
    # --- 数据准备 ---
    print("📦 正在准备 make_moons 数据集...")
    X_train, X_test, y_train, y_test = prepare_data(n_samples=1000, noise=0.2)
    print(f"   训练集: {X_train.shape[0]} 个样本")
    print(f"   测试集: {X_test.shape[0]} 个样本")
    print(f"   特征维度: {X_train.shape[1]}")
    print()
    
    # --- 训练超参数 ---
    HIDDEN_SIZE = 64
    EPOCHS = 300
    BATCH_SIZE = 64
    
    # --- SGD 训练 ---
    print("🔥 开始 SGD 训练...")
    sgd_history, sgd_params = train(
        X_train, y_train, X_test, y_test,
        hidden_size=HIDDEN_SIZE, optimizer_type='sgd',
        lr=0.1, epochs=EPOCHS, batch_size=BATCH_SIZE,
        verbose=True
    )
    print()
    
    # --- Adam 训练 ---
    print("🔥 开始 Adam 训练...")
    adam_history, adam_params = train(
        X_train, y_train, X_test, y_test,
        hidden_size=HIDDEN_SIZE, optimizer_type='adam',
        lr=0.01, epochs=EPOCHS, batch_size=BATCH_SIZE,
        verbose=True
    )
    print()
    
    # --- 结果对比 ---
    print("=" * 60)
    print("📊 训练结果对比")
    print("=" * 60)
    
    final_sgd_train = sgd_history['train_acc'][-1]
    final_sgd_test = sgd_history['test_acc'][-1]
    final_adam_train = adam_history['train_acc'][-1]
    final_adam_test = adam_history['test_acc'][-1]
    
    print(f"  SGD  - 训练准确率: {final_sgd_train:.4f} | 测试准确率: {final_sgd_test:.4f}")
    print(f"  Adam - 训练准确率: {final_adam_train:.4f} | 测试准确率: {final_adam_test:.4f}")
    print()
    
    # 分析收敛速度
    sgd_target_epoch = None
    adam_target_epoch = None
    target_acc = 0.90
    for i, acc in enumerate(sgd_history['test_acc']):
        if acc >= target_acc:
            sgd_target_epoch = i + 1
            break
    for i, acc in enumerate(adam_history['test_acc']):
        if acc >= target_acc:
            adam_target_epoch = i + 1
            break
    
    print(f"  达到 {target_acc:.0%} 测试准确率所需 epoch:")
    print(f"    SGD:  {sgd_target_epoch if sgd_target_epoch else '未达到'}")
    print(f"    Adam: {adam_target_epoch if adam_target_epoch else '未达到'}")
    print()
    
    # --- 绘图 ---
    print("📈 正在生成可视化图表...")
    plot_results(sgd_history, adam_history)
    
    # 绘制决策边界
    plot_decision_boundary(
        np.vstack([X_train, X_test]),
        np.concatenate([y_train, y_test]),
        adam_params,
        title="Adam 决策边界"
    )


if __name__ == '__main__':
    main()
