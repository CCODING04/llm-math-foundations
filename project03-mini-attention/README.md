# 🚀 阶段项目 3：Mini Attention + 解码采样器

> 🌱 *\"注意力机制是 Transformer 的心脏，采样策略是 LLM 的灵魂——前者决定模型怎么'看'，后者决定模型怎么'说'。\"*

### 📚 前置知识

完成本项目前，你需要掌握：
- **Ch7 注意力机制**：Q/K/V 的含义、Scaled Dot-Product Attention、多头注意力、因果掩码
- **Ch8 概率进阶与采样**：温度系数、Top-K 采样、Top-P（核采样）、Softmax 分布

---

## 🎯 项目目标

本项目的核心目标：**从零手写 Transformer 的注意力机制 + LLM 的解码策略，用纯 NumPy 实现，不依赖任何深度学习框架。**

| 完成后你将能够…… | |
|---|---|
| 用 NumPy 从零实现 Q/K/V 线性变换和注意力计算 | 🔍 |
| 手写 Scaled Dot-Product Attention 并理解每一步的 shape 变化 | 📐 |
| 实现因果掩码（Causal Mask）并理解其必要性 | 🎭 |
| 实现多头注意力的分头-计算-拼接流程 | 🧩 |
| 实现 Temperature / Top-K / Top-P 三种采样策略 | 🎲 |
| 对比不同采样参数对输出分布的影响 | 📊 |

**为什么用纯 NumPy？**

框架帮你做了太多事。`nn.MultiheadAttention` 一行代码就跑完了，但你知道里面发生了什么吗？手写一遍，你会真正理解：
- 为什么 Q/K/V 的权重矩阵 shape 是 `(d_model, d_k)`
- 为什么注意力分数要除以 `√d_k`
- 为什么因果掩码是一个下三角矩阵
- 多头注意力拼接后为什么要再做一次线性变换

---

## 📋 任务清单

| 步骤 | 任务 | 对应章节 |
|:---:|---|:---:|
| 1 | 实现 Q/K/V 线性变换 | Ch7 |
| 2 | 实现 Scaled Dot-Product Attention | Ch7 |
| 3 | 实现 Causal Mask（因果掩码） | Ch7 |
| 4 | 实现多头注意力拼接 | Ch7 |
| 5 | 实现 Temperature、Top-K、Top-P 三种采样策略 | Ch8 |
| 6 | 对比不同采样参数的输出差异 | Ch8 |

---

## 📝 步骤详解

### 步骤 1：实现 Q/K/V 线性变换

**目标**：将输入序列通过三个独立的线性变换，得到 Query、Key、Value 三个矩阵。

**数学公式**：

$$Q = X W_Q, \quad K = X W_K, \quad V = X W_V$$

其中：
- $X$：输入矩阵，shape = `(seq_len, d_model)`
- $W_Q, W_K, W_V$：权重矩阵，shape = `(d_model, d_k)`（或 `d_v`）
- $Q$：Query 矩阵，shape = `(seq_len, d_k)`
- $K$：Key 矩阵，shape = `(seq_len, d_k)`
- $V$：Value 矩阵，shape = `(seq_len, d_v)`

> 💡 **直观理解**：想象你在图书馆找书。Query 是"我想找什么"（你的搜索词），Key 是"每本书是什么"（书架标签），Value 是"书的内容"（实际信息）。注意力机制就是拿你的搜索词去和每本书的标签比，越匹配的书你越关注，最后返回的是内容的加权混合。

**代码片段**：

```python
import numpy as np

def linear_transform(X, W, b):
    """线性变换: Y = XW + b"""
    return X @ W + b

# 参数设置
seq_len = 4       # 序列长度
d_model = 8       # 模型维度
d_k = 6           # Q/K 的维度

# 随机初始化输入和权重
np.random.seed(42)
X = np.random.randn(seq_len, d_model)
W_Q = np.random.randn(d_model, d_k)
W_K = np.random.randn(d_model, d_k)
W_V = np.random.randn(d_model, d_k)
b_Q = np.zeros(d_k)
b_K = np.zeros(d_k)
b_V = np.zeros(d_k)

# Q/K/V 变换
Q = linear_transform(X, W_Q, b_Q)  # (4, 6)
K = linear_transform(X, W_K, b_K)  # (4, 6)
V = linear_transform(X, W_V, b_V)  # (4, 6)

print(f"X shape: {X.shape}")  # (4, 8)
print(f"Q shape: {Q.shape}")  # (4, 6)
print(f"K shape: {K.shape}")  # (4, 6)
print(f"V shape: {V.shape}")  # (4, 6)
```

---

### 步骤 2：实现 Scaled Dot-Product Attention

**目标**：计算注意力分数，加缩放，过 Softmax，加权求和 Value。

**数学公式**：

$$\text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

分步拆解：
1. **点积**：$S = QK^T$，shape = `(seq_len, seq_len)`，得到注意力分数矩阵
2. **缩放**：$S_{scaled} = S / \sqrt{d_k}$，防止分数过大导致 Softmax 梯度消失
3. **Softmax**：$\alpha = \text{Softmax}(S_{scaled})$，将分数变成概率分布
4. **加权求和**：$O = \alpha V$，shape = `(seq_len, d_v)`，得到输出

> ❓ **为什么要除以 √d_k？**
>
> 假设 Q 和 K 的每个分量都是均值为 0、方差为 1 的独立随机变量，那么点积 $q \cdot k = \sum_{i=1}^{d_k} q_i k_i$ 的方差就是 $d_k$。当 $d_k$ 很大时（比如 64），点积的值会非常大，Softmax 输出接近 one-hot，梯度接近 0——训练就"卡住"了。除以 $\sqrt{d_k}$ 把方差拉回 1，让 Softmax 的输入保持在合理范围。

**代码片段**：

```python
def softmax(x, axis=-1):
    """数值稳定的 Softmax"""
    x_max = np.max(x, axis=axis, keepdims=True)
    e_x = np.exp(x - x_max)
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def scaled_dot_product_attention(Q, K, V):
    """Scaled Dot-Product Attention"""
    d_k = Q.shape[-1]

    # 1. 点积 → 注意力分数矩阵
    scores = Q @ K.T              # (seq_len, seq_len)

    # 2. 缩放
    scores = scores / np.sqrt(d_k)

    # 3. Softmax → 注意力权重
    attn_weights = softmax(scores)  # (seq_len, seq_len)

    # 4. 加权求和
    output = attn_weights @ V       # (seq_len, d_v)

    return output, attn_weights

output, attn_weights = scaled_dot_product_attention(Q, K, V)
print(f"注意力权重:\n{attn_weights.round(3)}")
print(f"输出 shape: {output.shape}")  # (4, 6)
```

---

### 步骤 3：实现 Causal Mask（因果掩码）

**目标**：在自回归语言模型中，第 $i$ 个位置只能看到第 $0 \sim i$ 个位置，不能"偷看未来"。

**数学操作**：

$$S_{masked} = S + M, \quad M_{ij} = \begin{cases} 0 & \text{if } i \geq j \\ -\infty & \text{if } i < j \end{cases}$$

加上 $-\infty$ 后过 Softmax，未来位置的权重变成 $e^{-\infty} = 0$。

> 🎭 **直观理解**：你在考试时只能看已经做过的题，不能翻到后面偷看答案。因果掩码就是"禁止翻页"的规则。

**代码片段**：

```python
def create_causal_mask(seq_len):
    """创建因果掩码（下三角矩阵）"""
    mask = np.triu(np.full((seq_len, seq_len), -np.inf), k=1)
    return mask

def masked_attention(Q, K, V, mask=None):
    """带掩码的 Scaled Dot-Product Attention"""
    d_k = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)

    # 应用掩码：将未来位置的分数设为 -inf
    if mask is not None:
        scores = scores + mask

    attn_weights = softmax(scores)
    output = attn_weights @ V
    return output, attn_weights

causal_mask = create_causal_mask(seq_len)
output_masked, attn_masked = masked_attention(Q, K, V, mask=causal_mask)

print("因果掩码:")
print(causal_mask)
print("\n掩码后的注意力权重（上三角为 0）:")
print(attn_masked.round(3))
```

---

### 步骤 4：实现多头注意力拼接

**目标**：将注意力分成多个"头"，让模型同时关注不同子空间的信息。

**数学流程**：

1. **分头**：对每个头 $h$，分别做线性变换得到 $Q_h, K_h, V_h$
2. **独立计算**：每个头独立做 Scaled Dot-Product Attention
3. **拼接**：把所有头的输出拼接起来：$\text{Concat}(head_1, ..., head_H)$
4. **融合**：再过一个线性变换 $W_O$，得到最终输出

$$\text{MultiHead}(Q, K, V) = \text{Concat}(head_1, ..., head_H) W_O$$

其中 $head_h = \text{Attention}(QW_h^Q, KW_h^K, VW_h^V)$。

> 🧩 **直观理解**：想象你在读一篇文章。一个"头"关注语法结构，另一个关注情感色彩，第三个关注事实信息。多个头各司其职，最后汇总。

**代码片段**：

```python
def multi_head_attention(X, num_heads, d_model, d_k, d_v,
                          W_Qs, W_Ks, W_Vs, W_O, mask=None):
    """
    多头注意力
    - W_Qs: list of W_Q, 每个形状 (d_model, d_k)
    - W_O:  输出投影矩阵, 形状 (num_heads * d_v, d_model)
    """
    seq_len = X.shape[0]
    head_outputs = []

    for h in range(num_heads):
        Q_h = X @ W_Qs[h]
        K_h = X @ W_Ks[h]
        V_h = X @ W_Vs[h]

        out_h, _ = masked_attention(Q_h, K_h, V_h, mask=mask)
        head_outputs.append(out_h)

    # 拼接所有头的输出: (seq_len, num_heads * d_v)
    concat = np.concatenate(head_outputs, axis=-1)

    # 输出投影: (seq_len, d_model)
    output = concat @ W_O

    return output

# 初始化多头参数
num_heads = 2
d_k = d_v = d_model // num_heads  # 每个头的维度

W_Qs = [np.random.randn(d_model, d_k) for _ in range(num_heads)]
W_Ks = [np.random.randn(d_model, d_k) for _ in range(num_heads)]
W_Vs = [np.random.randn(d_model, d_v) for _ in range(num_heads)]
W_O  = np.random.randn(num_heads * d_v, d_model)

mha_output = multi_head_attention(
    X, num_heads, d_model, d_k, d_v,
    W_Qs, W_Ks, W_Vs, W_O, mask=causal_mask
)
print(f"多头注意力输出 shape: {mha_output.shape}")  # (4, 8)
```

---

### 步骤 5：实现 Temperature、Top-K、Top-P 采样策略

**目标**：模拟 LLM 解码时从概率分布中采样的过程，实现三种经典采样策略。

> 🎲 **为什么需要采样策略？** 模型输出的 logits 经过 Softmax 变成概率分布，但如何从这个分布中"选词"是大有学问的。不同的策略会影响生成文本的多样性、连贯性和创造性。

#### 5.1 Temperature 采样

**公式**：

$$P(w_i) = \frac{\exp(\text{logit}_i / T)}{\sum_j \exp(\text{logit}_j / T)}$$

- $T > 1$：分布更平坦，更随机，更有创意
- $T = 1$：标准 Softmax
- $T < 1$：分布更尖锐，更确定，更保守
- $T \to 0$：退化为贪心搜索

#### 5.2 Top-K 采样

**做法**：只保留概率最大的 K 个 token，其余设为 $-\infty$，再重新归一化。

- K 小（如 K=5）：输出更确定
- K 大（如 K=50）：输出更多样
- K=1：退化为贪心搜索

#### 5.3 Top-P（核采样 Nucleus Sampling）

**做法**：按概率从大到小排序，累积概率达到 P 时截断，只保留累积前 P 的 token。

- P 小（如 P=0.5）：输出更确定
- P 大（如 P=0.95）：输出更多样
- P=1.0：等价于标准随机采样

**代码片段**：

```python
def temperature_sampling(logits, temperature=1.0):
    """温度采样"""
    scaled_logits = logits / temperature
    probs = softmax(scaled_logits)
    return np.random.choice(len(probs), p=probs)

def top_k_sampling(logits, k=10):
    """Top-K 采样"""
    # 找到 top-k 的索引
    top_k_indices = np.argsort(logits)[-k:]
    # 非 top-k 位置设为 -inf
    filtered = np.full_like(logits, -np.inf)
    filtered[top_k_indices] = logits[top_k_indices]
    probs = softmax(filtered)
    return np.random.choice(len(probs), p=probs)

def top_p_sampling(logits, p=0.9):
    """Top-P（核采样）"""
    sorted_indices = np.argsort(logits)[::-1]
    sorted_logits = logits[sorted_indices]

    # 计算累积概率
    sorted_probs = softmax(sorted_logits)
    cumulative_probs = np.cumsum(sorted_probs)

    # 找到累积超过 p 的位置
    cutoff = np.searchsorted(cumulative_probs, p) + 1

    # 非前 cutoff 位置设为 -inf
    filtered = np.full_like(logits, -np.inf)
    filtered[sorted_indices[:cutoff]] = logits[sorted_indices[:cutoff]]

    probs = softmax(filtered)
    return np.random.choice(len(probs), p=probs)
```

---

### 步骤 6：对比不同采样参数的输出差异

**目标**：用同一组 logits，分别用不同参数采样多次，观察输出分布的差异。

**实验设计**：

```python
# 模拟一组 logits（假设词汇表大小 = 10）
logits = np.array([3.0, 2.5, 2.0, 1.5, 1.0, 0.5, 0.2, 0.1, 0.05, 0.01])

# 不同温度
for T in [0.1, 0.5, 1.0, 2.0, 5.0]:
    probs = softmax(logits / T)
    print(f"T={T:<4} → {probs.round(3)}")

# 不同 Top-K
for k in [1, 3, 5, 10]:
    # ... 对比输出分布

# 不同 Top-P
for p in [0.3, 0.5, 0.9, 1.0]:
    # ... 对比输出分布
```

**预期结果**：
- **低温度**（T=0.1）：几乎只在最大概率 token 上采样，输出确定
- **高温度**（T=5.0）：分布接近均匀，输出随机
- **小 K**（K=3）：只从前 3 个 token 中选
- **小 P**（P=0.3）：只从概率累积 30% 的 token 中选
- **K=1 或 P→0**：退化为贪心，永远选最大概率 token

---

## 📊 知识点回顾

| 概念 | 公式 / 关键点 | 出现章节 |
|---|---|:---:|
| Q/K/V 变换 | $Q=XW_Q, K=XW_K, V=XW_V$ | Ch7 |
| Scaled Dot-Product | $\text{softmax}(QK^T / \sqrt{d_k}) V$ | Ch7 |
| 缩放因子 $\sqrt{d_k}$ | 让点积方差回到 1，防止梯度消失 | Ch7 |
| Causal Mask | 上三角加 $-\infty$，防止看到未来 | Ch7 |
| 多头注意力 | 分头计算 → 拼接 → 线性投影 $W_O$ | Ch7 |
| Softmax | $\text{softmax}(z_i) = e^{z_i} / \sum_j e^{z_j}$，输出概率分布 | Ch7 |
| Temperature | 除以 T：T↑ 更随机，T↓ 更确定 | Ch8 |
| Top-K | 只保留概率最大的 K 个 token | Ch8 |
| Top-P（核采样） | 累积概率达到 P 时截断 | Ch8 |
| 数值稳定 Softmax | 先减最大值 $\max(z)$ 再算 $e^{z-\max}$ | Ch7 |

---

## 🏋️ 扩展挑战

### 挑战 1：实现完整的 Transformer Block

在多头注意力的基础上，加上：
- **Layer Normalization**：$\text{LN}(x) = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \cdot \gamma + \beta$
- **残差连接**：$x_{out} = x + \text{MultiHeadAttention}(x)$
- **Feed-Forward Network**：两层全连接 + ReLU 激活

把这三个组件拼成一个完整的 Transformer Block，验证输入输出 shape 不变。

### 挑战 2：实现 KV Cache 加速推理

在自回归生成中，每一步只新增一个 token。之前已经算过的 K 和 V 不需要重复计算——把它们缓存起来：
- 第一次：计算完整的 Q/K/V，缓存 K₀、V₀
- 第二次：只算新的 $q_1$，复用 $K_0$、$V_0$，拼接新 $k_1$、$v_1$
- 观察有/无 KV Cache 时，生成长度为 N 的序列分别需要多少次矩阵乘法

### 挑战 3：Beam Search 解码

实现 Beam Search（束搜索），维护 beam_width 条候选序列：
- 每一步：对每条候选序列扩展所有可能的下一个 token
- 计算每条扩展序列的累积对数概率
- 只保留概率最高的 beam_width 条
- 最终选择总概率最大的序列

对比 Beam Search 和贪心搜索在模拟 logits 上的输出差异。

---

## 📂 文件结构

```
project03-mini-attention/
├── README.md                      # 项目说明（你正在看的这个文件）
└── scripts/
    └── mini_attention.py          # 完整可运行脚本
```

### 运行方式

```bash
cd project03-mini-attention
python scripts/mini_attention.py
```

---

<p align="center">
  <i>🌱 手写一遍，胜过看十遍。注意力机制的每个细节，都藏在这些代码里。</i>
</p>
