# 🚀 阶段项目 2：从零实现两层神经网络

> **前置知识**：需要完成 Ch4（矩阵求导）+ Ch5（优化理论）+ Ch6（信息论）
>
> 🌱 **一句话概括**：把 Ch4-Ch6 学到的矩阵求导、交叉熵、优化器，全部串起来——手写一个能真正分类的神经网络。

---

## 🎯 项目目标

学完 Ch4-Ch6，你已经掌握了三个关键工具：

| 章节 | 学到了什么 | 本项目用在哪 |
|:---:|:---|:---|
| Ch4 | 矩阵求导、链式法则 | 反向传播：计算 ∂L/∂W₁ 和 ∂L/∂W₂ |
| Ch5 | 梯度下降、SGD、Adam | 手写优化器更新参数 |
| Ch6 | 交叉熵、信息熵 | 损失函数：衡量预测和真实的差距 |

本项目的目标就是**把这三块拼在一起**，从零实现一个两层神经网络，在 `make_moons` 数据集上完成二分类任务。

```
你将构建的网络：

输入 X (N×2) 
    → 全连接 + ReLU → 隐藏层 h (N×H)
    → 全连接 + Softmax → 输出 y_hat (N×2)
    → 交叉熵损失 → 反向传播 → Adam 更新参数
    → 循环 N 轮 → 得到一个能分类的模型！
```

---

## 📋 任务清单

### 任务 1：手写前向传播

**目标**：实现 `h = ReLU(XW₁ + b₁)` 和 `y_hat = softmax(hW₂ + b₂)`

前向传播就是数据的「正向流动」——输入经过两层变换，输出预测概率。

**第一层（隐藏层）**：
```python
z1 = X @ W1 + b1       # 线性变换：(N, D) @ (D, H) + (H,) → (N, H)
h = relu(z1)            # 激活函数：max(0, z)，引入非线性
```

**为什么需要 ReLU？** 如果只有线性变换，两层网络等价于一层（$W_2(W_1 X) = (W_2 W_1) X$）。ReLU 引入非线性，让网络能学习复杂模式。

**第二层（输出层）**：
```python
z2 = h @ W2 + b2        # 线性变换：(N, H) @ (H, C) + (C,) → (N, C)
y_hat = softmax(z2)     # 转换为概率分布：每行求和 = 1
```

**Softmax 公式**：
$$\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_j e^{z_j}}$$

> 💡 **数值稳定技巧**：先减去最大值再算 exp，防止数值溢出。

```python
def softmax(z):
    z_shifted = z - np.max(z, axis=1, keepdims=True)  # 减最大值
    exp_z = np.exp(z_shifted)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)
```

---

### 任务 2：手写交叉熵损失

**目标**：实现 $L = -\frac{1}{N} \sum_{i} y_i \log(\hat{y}_i)$

交叉熵（Ch6 核心概念）衡量预测分布 $\hat{y}$ 和真实分布 $y$ 之间的差异。**差异越大，损失越大**。

```python
def cross_entropy_loss(y_hat, y_onehot):
    N = y_hat.shape[0]
    y_hat_clipped = np.clip(y_hat, 1e-12, 1 - 1e-12)  # 防 log(0)
    loss = -np.sum(y_onehot * np.log(y_hat_clipped)) / N
    return loss
```

**为什么用交叉熵而不是均方误差？**
- 交叉熵的梯度 $\hat{y} - y$ 在预测错误时很大，预测正确时接近 0——**学习信号清晰**
- 均方误差配合 softmax 会导致梯度消失问题（饱和区梯度极小）

---

### 任务 3：手写反向传播

**目标**：用矩阵求导（Ch4）计算 $\frac{\partial L}{\partial W_1}$, $\frac{\partial L}{\partial b_1}$, $\frac{\partial L}{\partial W_2}$, $\frac{\partial L}{\partial b_2}$

这是本项目最核心的部分——把 Ch4 学的链式法则用起来。

#### 推导过程

**步骤 1：输出层梯度**

交叉熵 + Softmax 的组合有一个优美的性质，它们的梯度合在一起非常简洁：

$$\frac{\partial L}{\partial z_2} = \hat{y} - y \quad \text{(shape: N×C)}$$

> 🤔 **为什么这么简单？** 这是 softmax 的指数性质和交叉熵的对数性质「抵消」的结果。具体推导见 Ch6 的「交叉熵损失」一节。

```python
dz2 = y_hat - y_onehot   # (N, C)
```

**步骤 2：第二层参数梯度**

$$\frac{\partial L}{\partial W_2} = \frac{1}{N} h^T \cdot \frac{\partial L}{\partial z_2}$$

```python
dW2 = h.T @ dz2 / N     # (H, C)
db2 = np.sum(dz2, axis=0) / N  # (C,)
```

**步骤 3：传播到隐藏层**

$$\frac{\partial L}{\partial h} = \frac{\partial L}{\partial z_2} \cdot W_2^T$$

$$\frac{\partial L}{\partial z_1} = \frac{\partial L}{\partial h} \odot \text{ReLU}'(z_1)$$

```python
dh = dz2 @ W2.T                   # (N, H)
dz1 = dh * relu_derivative(z1)    # 逐元素乘法
```

**步骤 4：第一层参数梯度**

$$\frac{\partial L}{\partial W_1} = \frac{1}{N} X^T \cdot \frac{\partial L}{\partial z_1}$$

```python
dW1 = X.T @ dz1 / N     # (D, H)
db1 = np.sum(dz1, axis=0) / N  # (H,)
```

#### 梯度 shape 一览

| 梯度 | 公式 | Shape |
|:---|:---|:---|
| ∂L/∂z2 | y_hat - y | (N, C) |
| ∂L/∂W2 | h^T @ dz2 / N | (H, C) |
| ∂L/∂b2 | sum(dz2, axis=0) / N | (C,) |
| ∂L/∂h | dz2 @ W2^T | (N, H) |
| ∂L/∂z1 | dh ⊙ ReLU'(z1) | (N, H) |
| ∂L/∂W1 | X^T @ dz1 / N | (D, H) |
| ∂L/∂b1 | sum(dz1, axis=0) / N | (H,) |

> 💡 **检查梯度 shape 的技巧**：∂L/∂W 的 shape 一定和 W 相同。这是快速验证反向传播代码是否正确的第一道关卡。

---

### 任务 4：用 SGD 和 Adam 分别训练

**目标**：手写 SGD（带动量）和 Adam 优化器，对比收敛速度

#### SGD（带动量）

```python
class SGD:
    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.velocities = {}
    
    def step(self, params, grads):
        for key in params:
            if key not in self.velocities:
                self.velocities[key] = np.zeros_like(params[key])
            v = self.momentum * self.velocities[key] - self.lr * grads[key]
            self.velocities[key] = v
            params[key] = params[key] + v
```

动量的直觉：就像一个球在山坡上滚动，积累了「惯性」，能冲过小的坑洼。

#### Adam（手写实现）

Adam = Momentum + RMSprop，是当前最流行的优化器。

```python
class Adam:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = {}    # 一阶矩（动量方向）
        self.v = {}    # 二阶矩（自适应学习率）
        self.t = 0     # 时间步
    
    def step(self, params, grads):
        self.t += 1
        for key in params:
            g = grads[key]
            # 一阶矩更新
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * g
            # 二阶矩更新
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * g**2
            # 偏差修正
            m_hat = self.m[key] / (1 - self.beta1 ** self.t)
            v_hat = self.v[key] / (1 - self.beta2 ** self.t)
            # 参数更新
            params[key] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
```

**Adam 的关键创新**：
- **自适应学习率**：每个参数有自己的学习率（由二阶矩 v 控制），梯度大的参数步子小，梯度小的参数步子大
- **偏差修正**：初始时 m 和 v 都偏向 0，除以 $(1 - \beta^t)$ 修正这个偏差

**预期结果**：Adam 通常比 SGD 收敛更快、更稳定。

---

### 任务 5：画出 loss 曲线、准确率曲线

**目标**：可视化训练过程，直观感受 SGD 和 Adam 的区别

完整代码会生成：
1. **Loss 曲线**：训练集和测试集的交叉熵损失随 epoch 的变化
2. **准确率曲线**：训练集和测试集准确率随 epoch 的变化
3. **决策边界**：模型在特征空间中的分类区域

```bash
# 运行完整脚本
cd project02-two-layer-nn/scripts/
python two_layer_nn.py
```

预期输出类似：
```
📊 训练结果对比
  SGD  - 训练准确率: 0.87XX | 测试准确率: 0.86XX
  Adam - 训练准确率: 0.95XX | 测试准确率: 0.94XX

  达到 90% 测试准确率所需 epoch:
    SGD:  未达到（或较晚）
    Adam: 约 XX epoch（明显更快）
```

---

## 📝 知识点回顾

| 步骤 | 涉及章节 | 核心知识点 |
|:---:|:---:|:---|
| 前向传播 | Ch1, Ch4 | 矩阵乘法、广播、ReLU、Softmax |
| 交叉熵损失 | Ch6 | 信息熵、交叉熵、最大似然估计 |
| 反向传播 | Ch4 | 矩阵求导、链式法则、转置规则 |
| SGD 优化 | Ch5 | 梯度下降、学习率、动量 |
| Adam 优化 | Ch5 | 自适应学习率、一阶/二阶矩、偏差修正 |
| 参数初始化 | Ch4 | He 初始化（适用于 ReLU） |
| 可视化 | — | Loss 曲线、决策边界、过拟合判断 |

---

## 🔥 扩展挑战

完成基础任务后，试试这些扩展：

### 挑战 1：添加 L2 正则化

在损失函数中加入权重衰减（L2 正则化）：

$$L_{\text{total}} = L_{\text{CE}} + \lambda \left(\|W_1\|_F^2 + \|W_2\|_F^2\right)$$

**提示**：对应的梯度变化是 $\frac{\partial L}{\partial W} += \frac{2\lambda}{N} W$。

观察不同的 $\lambda$（如 0.001, 0.01, 0.1）如何影响训练和测试准确率的差距。

### 挑战 2：学习率调度

实现**余弦退火（Cosine Annealing）**学习率调度：

$$\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})(1 + \cos(\frac{t}{T}\pi))$$

对比固定学习率和余弦退火的效果。**提示**：在 Adam 的 `step()` 中加入 `self.lr = self.base_lr * ...`。

### 挑战 3：多层网络

将两层网络扩展为**三层**（加入第二个隐藏层）：

```
X → ReLU(XW₁+b₁) → ReLU(h₁W₂+b₂) → softmax(h₂W₃+b₃) → y_hat
```

你需要：
1. 添加第三组参数 $W_3, b_3$
2. 修改前向传播
3. 修改反向传播（多一层链式法则）
4. 观察更深的网络是否提升性能

---

## 🏃 快速开始

```bash
# 安装依赖
pip install numpy matplotlib scikit-learn

# 运行完整脚本
python scripts/two_layer_nn.py
```

---

## 📂 文件结构

```
project02-two-layer-nn/
├── README.md                      # 项目说明（你正在看的这个文件）
└── scripts/
    └── two_layer_nn.py            # 完整可运行脚本
```

---

<p align="center">
  <i>🌱 纸上得来终觉浅，绝知此事要躬行。把 Ch4-Ch6 的知识转化为代码，才是真正的理解。</i>
</p>
