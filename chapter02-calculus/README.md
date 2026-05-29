# 第二章：微积分核心 —— 变化的语言

> *「微积分是研究变化的数学。而训练一个大语言模型，本质上就是一场关于"变化"的旅程——找到让损失函数下降最快的方向，然后一步步走过去。」*

---

## 🎯 本章目标

| 学完这一章，你将能够… | 对应 LLM 场景 |
|---|---|
| 理解导数和偏导数的定义与计算 | 理解梯度下降中每个权重如何更新 |
| 掌握链式法则的推导过程 | 理解反向传播的数学本质 |
| 理解梯度的几何含义 | 理解为什么梯度指向"最快上升"方向 |
| 了解积分的基本概念 | 理解概率分布中的面积与期望 |

**在 LLM 中的位置**：训练 GPT 的核心循环是——

```
前向传播（算预测） → 算损失 → 反向传播（用链式法则求梯度） → 梯度下降（更新权重）
```

这一章我们解锁的就是 **反向传播 + 梯度下降** 的数学基础。

---

## 2.1 函数与极限 —— 微积分的起点

### 什么是函数？

函数就像一台「加工机器」：你丢进去一个数 $x$，它吐出来另一个数 $y$。

$$y = f(x)$$

在 LLM 中，到处都是函数：
- 线性变换：$z = Wx + b$
- 激活函数：$a = \text{ReLU}(z) = \max(0, z)$
- 损失函数：$L = \frac{1}{N}\sum(y_{\text{pred}} - y_{\text{true}})^2$

### 极限：微积分的地基

**极限**描述的是：当 $x$ 越来越接近某个值 $a$ 时，$f(x)$ 会趋向什么？

$$\lim_{x \to a} f(x) = L$$

意思是：$x$ 可以无限接近 $a$（但不一定等于 $a$），此时 $f(x)$ 无限接近 $L$。

**一个直观的例子**：

$$\lim_{x \to 0} \frac{\sin x}{x} = 1$$

当 $x$ 越来越接近 0 时，$\frac{\sin x}{x}$ 越来越接近 1。注意：$x=0$ 时这个式子是 $\frac{0}{0}$（无定义），但极限存在！

> 💡 **为什么极限重要？** 因为导数的定义就建立在极限之上——它是微积分最底层的砖块。

有了极限这个工具，接下来我们就可以定义导数了——它本质上就是一个特殊的极限。

---

## 2.2 导数 —— 瞬间变化的度量

### 2.2.1 从"平均变化率"到"瞬间变化率"

想象你开车从上海到杭州，300 公里花了 3 小时，平均速度 100 km/h。但途中你肯定有时快有时慢。那在某个**特定瞬间**，你的速度是多少？

这就是导数的直觉：**瞬时变化率**。

**平均变化率**（过两点的割线斜率）：

$$\frac{\Delta y}{\Delta x} = \frac{f(x + \Delta x) - f(x)}{\Delta x}$$

**导数**（让 $\Delta x$ 趋近于 0）：

$$f'(x) = \lim_{\Delta x \to 0} \frac{f(x + \Delta x) - f(x)}{\Delta x}$$

> 🎨 **比喻**：割线是"用尺子连两点"看斜率，导数是"无限放大曲线，直到看起来像直线"——这条直线的斜率就是导数。
>
> ![导数的几何意义：从割线到切线](./images/derivative_tangent.png)

### 2.2.2 几何意义：切线的斜率

导数 $f'(x)$ 就是函数在点 $x$ 处**切线的斜率**。

- $f'(x) > 0$：函数在上升 ↗️
- $f'(x) < 0$：函数在下降 ↘️
- $f'(x) = 0$：函数在平缓处，可能是极值点 ⬆️ 或 ⬇️

**在 LLM 中**：损失函数的导数告诉我们——「如果参数往这个方向微调一点点，损失会增大还是减小？」

### 2.2.3 物理意义：瞬时变化率

如果 $f(t)$ 表示时间 $t$ 时的位置，那么 $f'(t)$ 就是速度，$f''(t)$ 就是加速度。

### 2.2.4 基本求导公式（记住这些就够了）

| 函数 $f(x)$ | 导数 $f'(x)$ | 记忆口诀 |
|---|---|---|
| $c$（常数） | $0$ | 常数不变，变化率为零 |
| $x^n$ | $nx^{n-1}$ | 「降幂乘前」 |
| $e^x$ | $e^x$ | 唯一等于自己导数的函数 |
| $\ln x$ | $\frac{1}{x}$ | 自然对数的导数 |
| $\sin x$ | $\cos x$ | 正弦余弦互为导数 |
| $\cos x$ | $-\sin x$ | 注意负号！ |

### 2.2.5 求导法则

**加法法则**：
$$(f + g)' = f' + g'$$

**乘法法则**（莱布尼茨法则）：
$$(fg)' = f'g + fg'$$
> 记忆：「前导后不导 + 前不导后导」

**除法法则**：
$$\left(\frac{f}{g}\right)' = \frac{f'g - fg'}{g^2}$$

### 📝 例题：一步步求导

**求 $f(x) = 3x^4 - 2x^2 + 5x - 7$ 的导数**

$$f'(x) = 3 \cdot 4x^3 - 2 \cdot 2x + 5 \cdot 1 - 0 = 12x^3 - 4x + 5$$

就这么简单——逐项用「降幂乘前」法则即可。

**求 $f(x) = e^x \sin x$ 的导数**（乘法法则）

$$f'(x) = (e^x)'\sin x + e^x(\sin x)' = e^x\sin x + e^x\cos x = e^x(\sin x + \cos x)$$

---

## 2.3 偏导数 —— 多变量世界的变化率

### 问题来了：函数有多个变量怎么办？

在神经网络中，损失函数 $L$ 依赖于成千上万个参数 $w_1, w_2, \ldots, w_n$。我们想知道：**只改变其中一个参数，损失怎么变？**

### 定义

对于函数 $f(x, y)$，对 $x$ 的偏导数：

$$\frac{\partial f}{\partial x} = \lim_{\Delta x \to 0} \frac{f(x + \Delta x, y) - f(x, y)}{\Delta x}$$

关键：**把其他变量当作常数，只对一个变量求导**。

> 🎨 **比喻**：想象你站在一座山上（高度 = $f(x,y)$），偏导数就是——你只往东走（改变 $x$），不看南北（固定 $y$），脚下的坡度是多少？

### 📝 例题

**$f(x, y) = 3x^2y + y^3$，求 $\frac{\partial f}{\partial x}$ 和 $\frac{\partial f}{\partial y}$**

对 $x$ 求偏导（把 $y$ 当常数）：
$$\frac{\partial f}{\partial x} = 6xy$$

对 $y$ 求偏导（把 $x$ 当常数）：
$$\frac{\partial f}{\partial y} = 3x^2 + 3y^2$$

### Sigmoid 函数

在神经网络中经常出现的 sigmoid 函数定义为：

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

把任意数压到 0~1 之间的函数，叫 **sigmoid 函数**。它在神经网络中用作激活函数，输出可以解释为概率。

### 在神经网络中

一个简单的神经元：$z = w_1 x_1 + w_2 x_2 + b$，激活后 $a = \sigma(z)$，损失 $L = (a - y)^2$

要更新 $w_1$，我们需要的就是 $\frac{\partial L}{\partial w_1}$——这引出了我们的下一个主角：**链式法则**。

---

## 2.4 链式法则 —— 反向传播的数学灵魂 ⭐⭐⭐

> 如果这一章只记住一个知识点，**就是这个**。

### 2.4.1 为什么需要链式法则？

很多函数是**嵌套**的：$y = f(g(x))$。比如在神经网络中：

$$L = \text{Loss}(\sigma(w \cdot x + b))$$

损失是激活值的函数，激活值是线性组合的函数，线性组合又是权重的函数。一层套一层。

**链式法则**告诉我们怎么求这种复合函数的导数。

### 2.4.2 单变量链式法则

如果 $y = f(g(x))$，令 $u = g(x)$，则：

$$\frac{dy}{dx} = \frac{dy}{du} \cdot \frac{du}{dx}$$

> 🎨 **比喻**：就像齿轮传动——大齿轮转一圈，中齿轮转 3 圈；中齿轮转一圈，小齿轮转 5 圈。那么大齿轮转一圈，小齿轮转 $3 \times 5 = 15$ 圈。链式法则就是「逐级传递」。

### 📝 一步步推导

**求 $y = \sin(e^x)$ 的导数**

分解：
- $u = e^x$ → $\frac{du}{dx} = e^x$
- $y = \sin(u)$ → $\frac{dy}{du} = \cos(u) = \cos(e^x)$

链式法则：
$$\frac{dy}{dx} = \frac{dy}{du} \cdot \frac{du}{dx} = \cos(e^x) \cdot e^x$$

### 2.4.3 多变量链式法则

对于 $L = f(g(x, y), h(x, y))$，令 $u = g(x,y)$，$v = h(x,y)$：

$$\frac{\partial L}{\partial x} = \frac{\partial L}{\partial u}\frac{\partial u}{\partial x} + \frac{\partial L}{\partial v}\frac{\partial v}{\partial x}$$

### 📝 例题：多变量链式法则

**设 $L = (3x + 2y)^2$，求 $\frac{\partial L}{\partial x}$**

令 $u = 3x + 2y$，则 $L = u^2$。

由多变量链式法则：
$$\frac{\partial L}{\partial x} = \frac{\partial L}{\partial u} \cdot \frac{\partial u}{\partial x} = 2u \cdot 3 = 2(3x + 2y) \cdot 3 = 6(3x + 2y)$$

验证：展开 $L = 9x^2 + 12xy + 4y^2$，直接求偏导 $\frac{\partial L}{\partial x} = 18x + 12y = 6(3x + 2y)$。✅ 一致！

### 2.4.4 链式法则与反向传播 🧠

这就是反向传播的数学基础！看一个简单的前向传播：

```
输入 x → [z = wx + b] → [a = ReLU(z)] → [L = (a - y)²]
         线性层          激活函数           损失函数
```

**前向传播**（从左到右计算）：
1. $z = wx + b$
2. $a = \text{ReLU}(z) = \max(0, z)$
3. $L = (a - y)^2$

**反向传播**（从右到左求导，用链式法则）：
1. $\frac{\partial L}{\partial a} = 2(a - y)$
2. $\frac{\partial L}{\partial z} = \frac{\partial L}{\partial a} \cdot \frac{\partial a}{\partial z} = 2(a - y) \cdot \mathbb{1}_{z>0}$（其中 $\mathbb{1}_{z>0}$ 是指示函数，$z>0$ 时等于 1，否则等于 0）
3. $\frac{\partial L}{\partial w} = \frac{\partial L}{\partial z} \cdot \frac{\partial z}{\partial w} = 2(a - y) \cdot \mathbb{1}_{z>0} \cdot x$

最后一步就是用来更新权重 $w$ 的梯度！

> 💡 **关键洞察**：反向传播 = 链式法则的逐层应用，从输出往回「传递」导数。每一层只需要知道「上一层传来的梯度」和「本层的局部导数」，就能算出本层参数的梯度。

---

## 2.5 梯度 —— 指引方向的指南针

### 定义

梯度是把所有偏导数收集在一起形成的**向量**：

$$\nabla f = \left(\frac{\partial f}{\partial x_1}, \frac{\partial f}{\partial x_2}, \ldots, \frac{\partial f}{\partial x_n}\right)$$

> 💡 $\nabla$（读作 nabla，纳布拉）是一个类似于倒三角的算子符号。

### 梯度的三个重要性质

1. **方向**：梯度指向函数值**增长最快**的方向
2. **大小**：梯度的模 $|\nabla f|$ 表示最陡的坡度
3. **反方向**：$-\nabla f$ 指向函数值**下降最快**的方向

> 🎨 **比喻**：想象你蒙着眼睛站在山上，想尽快下山。你的脚能感受到哪个方向最陡——那个方向的反方向（$-\nabla f$）就是你该走的方向。这就是**梯度下降**！
>
> ![梯度下降可视化](./images/gradient_descent.png)

### 梯度下降公式

$$w_{\text{new}} = w_{\text{old}} - \eta \cdot \nabla_w L$$

其中 $\eta$（读作「艾塔」）是**学习率**——你每一步迈多大。

- $\eta$ 太大：步子迈太大，跳过最低点，来回震荡 🏃‍♂️💨
- $\eta$ 太小：走得太慢，训练时间太长 🐢
- $\eta$ 刚好：稳步下降，收敛到最小值 ✅

> ![学习率的影响](./images/learning_rate.png)

### 📝 例题：手动执行一步梯度下降

$f(w_1, w_2) = w_1^2 + 2w_2^2$，当前 $(w_1, w_2) = (1, 1)$，学习率 $\eta = 0.1$

**第一步**：算梯度
$$\frac{\partial f}{\partial w_1} = 2w_1 = 2, \quad \frac{\partial f}{\partial w_2} = 4w_2 = 4$$
$$\nabla f = (2, 4)$$

**第二步**：更新
$$w_1^{\text{new}} = 1 - 0.1 \times 2 = 0.8$$
$$w_2^{\text{new}} = 1 - 0.1 \times 4 = 0.6$$

函数值从 $1 + 2 = 3$ 降到了 $0.64 + 0.72 = 1.36$。✅ 确实在下降！

重复这个过程很多次，就会收敛到 $(0, 0)$——最小值点。

---

## 2.6 积分基础 —— 累积的艺术

### 2.6.1 不定积分：导数的逆运算

如果说导数是「求变化率」，那积分就是「已知变化率，求原函数」。

$$\int f(x)\,dx = F(x) + C \quad \text{其中} \quad F'(x) = f(x)$$

$C$ 是任意常数，因为导数会「丢失」常数信息。

> 🎨 **比喻**：导数是「压缩」（从路程到速度），积分是「解压缩」（从速度回到路程），但解压缩时会丢失一个「初始位置」的信息，所以 $+C$。

### 基本积分公式

| 被积函数 $f(x)$ | 积分 $\int f(x)\,dx$ |
|---|---|
| $x^n$ | $\frac{x^{n+1}}{n+1} + C$ ($n \neq -1$) |
| $\frac{1}{x}$ | $\ln\|x\| + C$ |
| $e^x$ | $e^x + C$ |
| $\cos x$ | $\sin x + C$ |
| $\sin x$ | $-\cos x + C$ |

### 2.6.2 定积分：曲线下的面积

$$\int_a^b f(x)\,dx = F(b) - F(a)$$

这是**牛顿-莱布尼茨公式**——它把「求面积」转化为「求原函数在端点的差」。

> 🎨 **比喻**：定积分就像用无数个极窄的矩形去逼近曲线下的面积。当矩形的宽度趋近于零时，求和就变成了积分。
>
> ![定积分的可视化](./images/integral_area.png)

### 📝 例题：手算一个定积分

**计算 $\int_0^2 3x^2\,dx$**

**第一步**：找原函数。$\int 3x^2\,dx = x^3 + C$（因为 $(x^3)' = 3x^2$）。

**第二步**：代入上下限（牛顿-莱布尼茨公式）：
$$\int_0^2 3x^2\,dx = \left[x^3\right]_0^2 = 2^3 - 0^3 = 8$$

所以曲线 $y = 3x^2$ 在 $[0, 2]$ 下的面积是 8。 ✅

### 2.6.3 积分在概率中的应用

在 LLM 中，模型输出的本质是**概率分布**。积分和概率密不可分：

**概率密度函数（PDF）** 的性质：
$$\int_{-\infty}^{+\infty} p(x)\,dx = 1$$

概率密度函数曲线下的总面积等于 1——这就像是说「所有可能性的总和是 100%」。

**期望值**（Expected Value）：
$$E[X] = \int_{-\infty}^{+\infty} x \cdot p(x)\,dx$$

期望就是「加权平均」，权重就是概率密度。

**方差**（Variance）：
$$\text{Var}(X) = E[(X - \mu)^2] = \int_{-\infty}^{+\infty} (x - \mu)^2 p(x)\,dx$$

其中 $\mu$（读作「缪」）表示均值/期望。

在 LLM 训练中，损失函数本质上也是一种期望——我们对所有训练样本的损失取平均。

---

## 2.7 🔢 公式推导：不跳步

### 推导 1：为什么梯度指向最快上升方向？

> 🎨 **直觉铺垫**：两个向量点积最大时就是方向相同的时候。所以要想让 $\nabla f \cdot \mathbf{u}$ 最大，$\mathbf{u}$ 应该和 $\nabla f$ 方向一致。

设当前位置 $\mathbf{w}$，走一步 $\Delta\mathbf{w}$（方向向量 $\mathbf{u}$，$|\mathbf{u}|=1$，步长 $\alpha$）：

$$f(\mathbf{w} + \alpha\mathbf{u}) \approx f(\mathbf{w}) + \alpha \nabla f \cdot \mathbf{u}$$

要让函数增长最快，就要最大化 $\nabla f \cdot \mathbf{u}$。

由柯西-施瓦茨不等式：

$$\nabla f \cdot \mathbf{u} \leq |\nabla f| \cdot |\mathbf{u}| = |\nabla f|$$

等号成立条件：$\mathbf{u} = \frac{\nabla f}{|\nabla f|}$

**结论**：$\nabla f$ 方向就是函数增长最快的方向！所以 $-\nabla f$ 就是下降最快的方向。 $\blacksquare$

### 推导 2：反向传播中的链式法则（两层的网络）

网络结构：
$$z = W_1 x, \quad h = \sigma(z), \quad \hat{y} = W_2 h, \quad L = \frac{1}{2}(\hat{y} - y)^2$$

**求 $\frac{\partial L}{\partial W_1}$**（一步步，不跳步）：

**Step 1**：$\frac{\partial L}{\partial \hat{y}} = \hat{y} - y$

**Step 2**：$\frac{\partial L}{\partial h} = \frac{\partial L}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial h} = (\hat{y} - y) \cdot W_2$

**Step 3**：$\frac{\partial L}{\partial z} = \frac{\partial L}{\partial h} \cdot \frac{\partial h}{\partial z} = (\hat{y} - y) \cdot W_2 \cdot \sigma'(z)$

**Step 4**：$\frac{\partial L}{\partial W_1} = \frac{\partial L}{\partial z} \cdot \frac{\partial z}{\partial W_1} = [(\hat{y} - y) \cdot W_2 \cdot \sigma'(z)] \cdot x^T$

看，每一步都是链式法则：**上一层的梯度 × 本层的局部导数 = 本层的梯度**。这就是反向传播的核心模式！

---

## 2.8 💻 代码验证

### 2.8.1 数值验证导数

```python
import numpy as np

# 数值导数（用定义来算）
def numerical_derivative(f, x, h=1e-5):
    return (f(x + h) - f(x - h)) / (2 * h)

# 测试几个函数
print("=== 数值导数验证 ===")
print(f"x² 在 x=3 处的导数: 数值={numerical_derivative(lambda x: x**2, 3):.6f}, 解析=6.000000")
print(f"sin(x) 在 x=π/4 处的导数: 数值={numerical_derivative(np.sin, np.pi/4):.6f}, 解析={np.cos(np.pi/4):.6f}")
print(f"e^x 在 x=1 处的导数: 数值={numerical_derivative(np.exp, 1):.6f}, 解析={np.exp(1):.6f}")
print(f"ln(x) 在 x=2 处的导数: 数值={numerical_derivative(np.log, 2):.6f}, 解析={1/2:.6f}")
```

### 2.8.2 验证链式法则

```python
# y = sin(e^x), 链式法则说 dy/dx = cos(e^x) * e^x
def composite_func(x):
    return np.sin(np.exp(x))

def chain_rule_derivative(x):
    return np.cos(np.exp(x)) * np.exp(x)

x_test = 1.0
print("\n=== 链式法则验证 ===")
print(f"d/dx[sin(e^x)] 在 x={x_test}:")
print(f"  数值导数: {numerical_derivative(composite_func, x_test):.6f}")
print(f"  链式法则: {chain_rule_derivative(x_test):.6f}")
```

### 2.8.3 可视化脚本

所有可视化脚本都保存在 `scripts/` 目录下：

| 脚本 | 生成图片 | 说明 |
|---|---|---|
| `plot_derivative.py` | `images/derivative_tangent.png` | 导数几何意义：函数与切线 |
| `plot_gradient_descent.py` | `images/gradient_descent.png` | 梯度下降过程可视化 |
| `plot_learning_rate.py` | `images/learning_rate.png` | 不同学习率的影响 |
| `plot_integral.py` | `images/integral_area.png` | 定积分 = 曲线下面积 |

运行所有脚本：
```bash
cd /tmp/llm-math-foundations/chapter02-calculus
python scripts/plot_derivative.py
python scripts/plot_gradient_descent.py
python scripts/plot_learning_rate.py
python scripts/plot_integral.py
```

---

## 2.9 🎯 LLM 关联总结

| 微积分概念 | LLM 中的角色 | 具体体现 |
|---|---|---|
| **导数** | 衡量参数微小变化对损失的影响 | $\frac{\partial L}{\partial w}$ 决定权重更新方向 |
| **偏导数** | 多参数中，单独看每个参数的影响 | 每个权重都有自己的梯度 |
| **链式法则** | 反向传播的数学基础 | 逐层传递梯度，从 Loss 一直传到每一层的参数 |
| **梯度** | 所有偏导数组成的向量 | $\nabla_w L$ 指引所有权重同时更新 |
| **梯度下降** | 训练的核心算法 | $w \leftarrow w - \eta \nabla_w L$ |
| **积分** | 概率分布与期望 | softmax 输出的概率分布、连续概率的归一化 |

**训练一个大模型的过程，用微积分的语言说就是**：

> 在一个拥有数十亿维的参数空间里，沿着损失函数梯度的反方向，一步一步往下走，直到找到（足够好的）最低点。

---

## 2.10 ❓ 思考题

**Q1（概念题）**：为什么说「导数为 0 的点不一定是极值点」？举一个反例。

**Q2（计算题）**：求 $f(x, y) = x^3 + x^2y + xy^2 + y^3$ 在点 $(1, 2)$ 处的梯度 $\nabla f$。

**Q3（链式法则）**：设 $z = (2x + 1)^5$，用链式法则求 $\frac{dz}{dx}$。如果直接展开多项式再求导，结果一样吗？哪种方法更快？

**Q4（LLM 思考）**：在梯度下降中，如果我们不小心把更新公式写成了 $w_{\text{new}} = w_{\text{old}} + \eta \cdot \nabla_w L$（加号而不是减号），会发生什么？为什么？

**Q5（开放题）**：梯度下降只能找到局部最小值，不能保证找到全局最小值。在实际训练 LLM 时，为什么这个问题没有想象中那么严重？（提示：高维空间的性质）

---

> 📚 **下一章预告**：第三章我们将进入**线性代数**的世界——矩阵、向量、特征分解，这些是理解 Transformer 注意力机制的钥匙！
