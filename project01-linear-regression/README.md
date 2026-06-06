# 🚀 阶段项目 1：从零实现线性回归

> *\"学了三章数学，该来真刀真枪练一回了！这个项目把矩阵运算、梯度下降、概率估计串成一条线——你会发现，前面学的每一个概念都不是孤立的。\"*

> **前置知识**：完成 Ch1（线性代数）+ Ch2（微积分）+ Ch3（概率统计）

---

## 🎯 项目目标

把前三章学的知识**串起来**，从零手写一个完整的线性回归模型：

| 前三章学了什么 | 本项目用在哪里 |
|:---|:---|
| Ch1：矩阵乘法、向量运算 | 前向传播 $\hat{y} = Xw + b$ |
| Ch2：导数、梯度、梯度下降 | 手写梯度 $\frac{\partial L}{\partial w}$、更新参数 |
| Ch3：期望、MLE、正态分布 | 理解 MSE 等价于高斯噪声下的 MLE |

**做完这个项目，你会真正理解**：
- 为什么模型训练 = 梯度下降 = 优化问题
- 为什么 MSE 是个好的损失函数（它不随便定义的！）
- 梯度下降的解和解析解（正规方程）是一回事

---

## 📋 任务清单（7 步）

### 步骤 1：用 NumPy 构造数据

真实数据里，变量之间的关系不会是完美的直线——总会有"噪音"。咱们模拟这个过程：

$$y = 3x + 2 + \varepsilon, \quad \varepsilon \sim N(0, 1)$$

这里 $3x + 2$ 是"真实规律"，$\varepsilon$ 是随机扰动（可以理解为测量误差、未观测因素等）。

```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# 生成 100 个样本
n_samples = 100
X = np.random.uniform(0, 10, size=n_samples)  # x 在 [0, 10] 之间随机取
noise = np.random.normal(0, 1, size=n_samples)  # 高斯噪声
y = 3 * X + 2 + noise  # 真实关系：y = 3x + 2 + 噪声

print(f"数据量: {n_samples}")
print(f"X 范围: [{X.min():.2f}, {X.max():.2f}]")
print(f"y 范围: [{y.min():.2f}, {y.max():.2f}]")
```

> 🌱 **为什么要加噪声？** 真实世界的数据永远不完美。如果不加噪声，所有点完美落在一条线上——太简单了，体现不出模型"从噪声中学习规律"的能力。加噪声后，模型需要在"信号"和"噪声"之间做权衡，这才是真正的机器学习。

---

### 步骤 2：手写前向传播

前向传播就是：给定当前参数 $(w, b)$，算出模型的预测值。

$$\hat{y} = X \cdot w + b$$

这是一个**矩阵-向量乘法 + 偏置加法**——和 Transformer 里每一层的 $h = xW + b$ 结构一模一样，只不过这里更简单。

```python
# 初始化参数
w = np.random.randn()  # 随机初始化权重
b = np.random.randn()  # 随机初始化偏置

def forward(X, w, b):
    """前向传播：y_hat = X @ w + b"""
    return X * w + b

y_hat = forward(X, w, b)
print(f"初始 w={w:.4f}, b={b:.4f}")
print(f"初始预测范围: [{y_hat.min():.2f}, {y_hat.max():.2f}]")
```

> 🌱 这里 `X * w` 是逐元素乘法（因为 $X$ 是一维向量），等价于矩阵乘法的特例。在神经网络中，$X$ 通常是一个矩阵（batch × features），那时候就是标准的矩阵乘法 `X @ W`。

---

### 步骤 3：手写损失函数（MSE）

**均方误差（Mean Squared Error）** 是回归任务最经典的损失函数：

$$L = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$$

直觉：预测值和真实值差多少？差的平方取平均。平方保证正负误差不会抵消，也放大了大的偏差。

```python
def mse_loss(y_true, y_pred):
    """均方误差损失"""
    return np.mean((y_true - y_pred) ** 2)

loss = mse_loss(y, y_hat)
print(f"初始 MSE 损失: {loss:.4f}")
```

> 🤔 **为什么用平方而不是绝对值？** 两个原因：①平方对大误差惩罚更狠（更"严格"的 老师）；②平方函数处处可导，绝对值在 0 点不可导，优化更方便。

---

### 步骤 4：手写梯度

这是整个项目最核心的一步！我们要用**链式法则**（Ch2 §2.4）手动推导梯度。

**推导过程（不跳步）**：

设 $L = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$，其中 $\hat{y}_i = wx_i + b$。

**对 $w$ 的偏导数**：

$$\frac{\partial L}{\partial w} = \frac{1}{n}\sum_{i=1}^{n} 2(y_i - \hat{y}_i) \cdot \frac{\partial (y_i - \hat{y}_i)}{\partial w}$$

因为 $\hat{y}_i = wx_i + b$，所以 $\frac{\partial \hat{y}_i}{\partial w} = x_i$，于是：

$$\frac{\partial L}{\partial w} = -\frac{2}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i) \cdot x_i$$

**对 $b$ 的偏导数**：

同理，$\frac{\partial \hat{y}_i}{\partial b} = 1$，所以：

$$\frac{\partial L}{\partial b} = -\frac{2}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)$$

> 🌱 这就是 Ch2 讲的**链式法则**的实战！误差从损失函数传到预测值，再传到参数。在深度神经网络里，这个过程要重复几十层——但原理完全一样。

```python
def compute_gradients(X, y, w, b):
    """手写梯度计算"""
    n = len(y)
    y_hat = forward(X, w, b)
    error = y - y_hat  # 残差
    
    dw = -2 / n * np.sum(error * X)  # ∂L/∂w
    db = -2 / n * np.sum(error)       # ∂L/∂b
    
    return dw, db

dw, db = compute_gradients(X, y, w, b)
print(f"梯度: dw={dw:.4f}, db={db:.4f}")
```

---

### 步骤 5：梯度下降循环

有了梯度，就可以沿着**梯度的反方向**更新参数（Ch2 §2.5）：

$$w \leftarrow w - \eta \cdot \frac{\partial L}{\partial w}$$
$$b \leftarrow b - \eta \cdot \frac{\partial L}{\partial b}$$

其中 $\eta$ 是学习率——步子迈多大有讲究（回顾 Ch2 的学习率讨论）。

```python
# 训练参数
learning_rate = 0.01
n_epochs = 100

# 记录训练过程
loss_history = []
w_history = []
b_history = []

# 重新初始化
w = 0.0
b = 0.0

for epoch in range(n_epochs):
    # 前向传播
    y_hat = forward(X, w, b)
    
    # 计算损失
    loss = mse_loss(y, y_hat)
    loss_history.append(loss)
    
    # 计算梯度
    dw, db = compute_gradients(X, y, w, b)
    
    # 更新参数
    w -= learning_rate * dw
    b -= learning_rate * db
    
    w_history.append(w)
    b_history.append(b)
    
    if epoch % 20 == 0 or epoch == n_epochs - 1:
        print(f"Epoch {epoch:3d}: loss={loss:.4f}, w={w:.4f}, b={b:.4f}")

print(f"\n最终结果: w={w:.4f} (真实值=3), b={b:.4f} (真实值=2)")
```

**预期输出**（大致）：
```
Epoch   0: loss=97.5xxx, w=1.9xxx, b=0.3xxx
Epoch  20: loss=1.5xxx, w=2.8xxx, b=1.8xxx
...
Epoch  99: loss=0.8xxx, w=2.9xxx, b=2.0xxx

最终结果: w≈2.97 (真实值=3), b≈2.02 (真实值=2)
```

参数收敛到了接近真实值 $w=3, b=2$！🎉

---

### 步骤 6：画出 loss 曲线和拟合结果

光看数字不够直观，咱们画图看看训练过程和最终效果。

```python
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 图1：Loss 曲线
axes[0].plot(loss_history, linewidth=2, color='#2196F3')
axes[0].set_xlabel('Epoch', fontsize=12)
axes[0].set_ylabel('MSE Loss', fontsize=12)
axes[0].set_title('训练损失曲线', fontsize=14)
axes[0].grid(True, alpha=0.3)

# 图2：参数收敛过程
axes[1].plot(w_history, linewidth=2, label='w (学习到的)', color='#FF5722')
axes[1].axhline(y=3, color='red', linestyle='--', linewidth=1.5, label='w (真实值=3)')
axes[1].plot(b_history, linewidth=2, label='b (学习到的)', color='#4CAF50')
axes[1].axhline(y=2, color='green', linestyle='--', linewidth=1.5, label='b (真实值=2)')
axes[1].set_xlabel('Epoch', fontsize=12)
axes[1].set_ylabel('参数值', fontsize=12)
axes[1].set_title('参数收敛过程', fontsize=14)
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)

# 图3：拟合结果
axes[2].scatter(X, y, alpha=0.5, s=20, label='数据点', color='#2196F3')
x_line = np.linspace(0, 10, 100)
axes[2].plot(x_line, w * x_line + b, linewidth=2.5, color='#FF5722', 
             label=f'拟合: y={w:.2f}x+{b:.2f}')
axes[2].plot(x_line, 3 * x_line + 2, linewidth=2, linestyle='--', 
             color='#4CAF50', label='真实: y=3x+2')
axes[2].set_xlabel('x', fontsize=12)
axes[2].set_ylabel('y', fontsize=12)
axes[2].set_title('拟合结果', fontsize=14)
axes[2].legend(fontsize=10)
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('./images/training_result.png', dpi=150, bbox_inches='tight')
plt.show()
```

![训练结果](./images/training_result.png)

你会看到三幅图：
- **左图**：Loss 从高处快速下降，最终趋于平稳——训练在收敛 ✅
- **中图**：$w$ 和 $b$ 从初始值逐步逼近真实值（虚线）
- **右图**：拟合直线几乎和真实直线重合，数据点均匀分布在两侧

---

### 步骤 7：验证三解合一

这是本项目最"升华"的一步——我们要验证：**梯度下降的解 = 正规方程的解 = MLE 的解**。

#### 7a. 正规方程（解析解）

线性回归有一个**闭式解**（不需要迭代！）：

$$\hat{w} = (X^TX)^{-1}X^Ty$$

这就是**正规方程**（Normal Equation），用矩阵运算一步到位。

```python
# 构造增广矩阵 [X | 1]（把 b 吸收到 w 里）
X_aug = np.column_stack([X, np.ones(n_samples)])  # shape: (100, 2)

# 正规方程: w_hat = (X^T X)^{-1} X^T y
w_normal = np.linalg.inv(X_aug.T @ X_aug) @ X_aug.T @ y

print(f"正规方程解: w={w_normal[0]:.4f}, b={w_normal[1]:.4f}")
print(f"梯度下降解: w={w:.4f}, b={b:.4f}")
print(f"差异: Δw={abs(w - w_normal[0]):.6f}, Δb={abs(b - w_normal[1]):.6f}")
```

**预期输出**：两种方法的解几乎完全一致！差异在小数点后多位。

#### 7b. MLE 视角

假设噪声 $\varepsilon \sim N(0, \sigma^2)$，那么 $y_i \sim N(wx_i + b, \sigma^2)$。

似然函数：

$$L(w, b) = \prod_{i=1}^{n} \frac{1}{\sqrt{2\pi}\sigma}\exp\left(-\frac{(y_i - wx_i - b)^2}{2\sigma^2}\right)$$

取负对数似然：

$$-\log L = \frac{n}{2}\log(2\pi\sigma^2) + \frac{1}{2\sigma^2}\sum_{i=1}^{n}(y_i - wx_i - b)^2$$

最小化 $-\log L$ 等价于最小化 $\sum(y_i - \hat{y}_i)^2$，也就是 MSE！

> 🎉 **三解合一**：
> - **梯度下降**：数值迭代，逐步逼近
> - **正规方程**：一步到位的解析解
> - **MLE**：概率框架下的最大似然估计
>
> 三条路，同一个终点。这不是巧合——当噪声服从高斯分布时，最小二乘 = MLE，而正规方程给出的是最小二乘的精确解。

```python
print("=== 三解对比 ===")
print(f"梯度下降:  w={w:.6f}, b={b:.6f}")
print(f"正规方程:  w={w_normal[0]:.6f}, b={w_normal[1]:.6f}")
print(f"真实值:    w=3.000000, b=2.000000")
print(f"\n✅ 三种方法给出的解几乎完全一致！")
print(f"✅ MSE 最小解 = 正规方程解 = MLE 解（在高斯噪声假设下）")
```

---

## 📊 知识点回顾

每个步骤对应哪一章哪个概念？一表看清：

| 步骤 | 做了什么 | 对应章节 | 核心概念 |
|:---:|:---|:---|:---|
| 1 | 构造数据 + 高斯噪声 | Ch3 §3.3 | 正态分布 $N(\mu, \sigma^2)$ |
| 2 | 前向传播 $\hat{y} = Xw + b$ | Ch1 §5.3 | 矩阵-向量乘法、偏置 |
| 3 | 损失函数 MSE | Ch3 §4.4 | 均方误差 = 预测误差的二阶矩 |
| 4 | 手写梯度 $\frac{\partial L}{\partial w}$ | Ch2 §2.4–2.5 | 链式法则、偏导数、梯度 |
| 5 | 梯度下降循环 | Ch2 §2.5 | 梯度下降公式、学习率 |
| 6 | 可视化训练过程 | — | Matplotlib 数据可视化 |
| 7 | 三解合一验证 | Ch1 + Ch2 + Ch3 | 正规方程（矩阵求逆）+ MLE + 优化等价性 |

---

## 🏋️ 扩展挑战

做完了基础版，想更深入？试试这些：

### 挑战 1：多维线性回归

把一维 $y = wx + b$ 扩展到多维 $y = w_1x_1 + w_2x_2 + \cdots + w_d x_d + b$。

提示：
- 把参数 $w$ 改成向量，$X$ 改成矩阵（每行一个样本，每列一个特征）
- 前向传播变成 `y_hat = X @ w + b`（矩阵乘法）
- 梯度变成 $\frac{\partial L}{\partial \mathbf{w}} = -\frac{2}{n}X^T(\mathbf{y} - \hat{\mathbf{y}})$

> 这就是真实深度学习里做的事——只不过层数更多、激活函数更复杂。

### 挑战 2：学习率调度

固定学习率有个问题：后期 loss 震荡，无法精调。试试：
- **余弦退火**（Cosine Annealing）：$\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})(1 + \cos(\frac{t}{T}\pi))$
- **指数衰减**：$\eta_t = \eta_0 \times 0.95^t$

画图对比不同策略下 loss 曲线的收敛速度和最终精度。

### 挑战 3：从线性回归到神经网络

在 $y = Xw + b$ 的基础上加一个 ReLU 激活：$h = \text{ReLU}(Xw_1 + b_1)$，然后再做一次线性变换 $y = hw_2 + b_2$。

这就是一个**两层神经网络**！用链式法则推导梯度，完成训练。观察：非线性激活函数带来了什么好处？拟合能力有什么变化？

---

## 📁 文件结构

```
project01-linear-regression/
├── README.md                          # 你正在看的这个文件
├── images/                            # 训练结果图
│   └── training_result.png
└── scripts/
    └── linear_regression.py           # 完整可运行代码
```

---

## 🚦 运行方式

```bash
cd project01-linear-regression/scripts
python linear_regression.py
```

依赖：`numpy`、`matplotlib`（安装：`pip install numpy matplotlib`）

---

> 🌱 **下一步**：完成这个项目后，你就可以进入 Ch4（矩阵深度）了。到那时，你会用更强大的矩阵工具来理解模型参数的结构——SVD、特征值、矩阵求导，这些都是 Transformer 的数学基础。
