# 第七章：注意力机制的数学

> 🌱 *"注意力机制是 Transformer 的心脏——它让模型学会'看哪里'。"*

---

## 🎯 本章目标 + 在 LLM 中的位置

### 本章学完你能做什么？

- ✅ 手推 Self-Attention 的完整计算过程，不跳任何一步
- ✅ 理解 Q/K/V 的数学含义，而不是把它们当成黑箱
- ✅ 解释为什么注意力分数要除以 √d_k（用方差的数学推导说服自己）
- ✅ 从零实现 Multi-Head Attention，并画出注意力热力图
- ✅ 理解因果掩码如何保证自回归生成的正确性
- ✅ 说清 KV Cache 为什么能加速推理

### 注意力在 LLM 中的位置

还记得咱们之前学过的架构吗？一个 Transformer 块长这样：

```
输入 x → [Layer Norm] → [Self-Attention] → 残差连接 → [Layer Norm] → [FFN] → 残差连接 → 输出
```

**Self-Attention 是 Transformer 唯一让不同位置的 token 相互交流的机制。** 没有 Attention，每个 token 就是一个孤岛——它只能看到自己，无法理解上下文。

> 💡 想象你在读一句话："苹果发布了新手机。" 你看到"苹果"这个词时，你需要往后看"手机"才能确定它是水果还是公司。**Attention 就是让"苹果"去"看"其他词的机制。**

---

## 📝 概念讲解

### 1. Softmax 函数

在讲注意力之前，咱们得先认识它的"好搭档"——Softmax。

#### 公式

对于一个向量 **z** = [z₁, z₂, ..., zₙ]，Softmax 定义为：

$$\text{Softmax}(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{n} e^{z_j}}$$

#### 为什么用 eˣ 而不是其他函数？

好问题！咱们来想想，把分数变成概率分布，需要满足什么条件？

1. **所有值都是正的** → eˣ 永远 > 0 ✅
2. **所有值加起来等于 1** → 因为分母是所有 eˣ 的和 ✅
3. **大的值应该得到更大的权重** → eˣ 是单调递增的 ✅
4. **可微分** → eˣ 的导数是 eˣ 自己 ✅

用 x² 行不行？不行——x² 在负数域不是单调的。用 |x| 呢？在 0 点不可微。所以 **eˣ 几乎是唯一满足所有条件的"天选之子"**。

#### 温度参数

在注意力中，Softmax 常常带一个温度参数 T：

$$\text{Softmax}(z_i, T) = \frac{e^{z_i / T}}{\sum_{j=1}^{n} e^{z_j / T}}$$

- T 大 → 分布更均匀（"我很犹豫，谁都差不多"）
- T 小 → 分布更尖锐（"我很确定，就选这个！"）
- T → 0 → 趋近 one-hot（"就是它了！"）
- T → ∞ → 趋近均匀分布（"随便吧"）

> 🎨 想象你在选餐厅。温度高 = "都行啊随便"，温度低 = "我只吃火锅！" LLM 生成的 diversity 也受温度控制。

#### 温度 T 和缩放因子 √d_k 的关系

你可能会问：温度 T 和前面提到的 √d_k 不都是做"缩放"吗？没错，但它们的**目的不同**：

- **√d_k（缩放因子）**：是固定的、不可调的，目的是**控制方差**，让 Softmax 的输入不会因为维度增大而爆炸，保证训练稳定。它解决的是"数学上的问题"。
- **T（温度参数）**：是可调的超参数，目的是**控制生成多样性**。T 大则输出更随机、更有创意；T 小则输出更确定、更保守。它解决的是"应用上的需求"。

> 💡 简单记：√d_k 是"安全带"（防事故），T 是"油门"（控速度）。

---

### 2. 缩放点积注意力（Scaled Dot-Product Attention）

这是 Attention 的核心公式：

$$\text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

别急，咱们一步步拆解。

#### Q、K、V 是什么？

- **Q（Query）**：我在找什么？——"我想知道这个词和其他词的关系"
- **K（Key）**：我有什么？——"我身上有什么信息可以被匹配"
- **V（Value）**：我的内容是什么？——"匹配上了就给你这些信息"

> 🖥️ 类比图书馆找书：你心里有个查询意图（Q），每本书有标签（K），找到匹配的书后你读它的内容（V）。

数学上，Q、K、V 是输入向量 **x** 经过线性变换得到的：

$$Q = x W_Q, \quad K = x W_K, \quad V = x W_V$$

其中 W_Q, W_K, W_V 是可学习的参数矩阵。**注意：这三个矩阵让模型学习"怎么问""怎么答""答什么"。**

#### 注意力分数计算（完整推导）

假设输入序列有 n 个 token，每个 token 的嵌入维度是 d_model。

> 💡 **d_model 是什么？** d_model 是模型的嵌入维度（embedding dimension），也就是每个 token 被表示成多长的向量。例如 GPT-2 的 d_model = 768，意味着每个 token 用 768 维的向量表示。维度越高，能编码的信息越丰富，但计算量也越大。

**Step 1: 计算注意力分数矩阵**

Q 的形状：[n, d_k]，K 的形状：[n, d_k]

$$S = QK^T \quad \text{形状：[n, n]}$$

矩阵 S 的第 i 行第 j 列的元素：s_ij = q_i · k_j（点积），表示 token i 对 token j 的"原始关注度"。

**Step 2: 缩放**

$$\hat{S} = \frac{S}{\sqrt{d_k}}$$

**为什么要除以 √d_k？** 这是个极好的数学问题，咱们来推导一下。

假设 q_i 和 k_j 的每个分量都是均值为 0、方差为 1 的独立随机变量。那么它们的点积：

$$s_{ij} = \sum_{l=1}^{d_k} q_{il} \cdot k_{jl}$$

因为各项独立，方差可以相加：

$$\text{Var}(s_{ij}) = \sum_{l=1}^{d_k} \text{Var}(q_{il} \cdot k_{jl}) = d_k \cdot \text{Var}(q \cdot k)$$

如果 Var(q) = Var(k) = 1，那么 Var(q·k) = 1，所以：

$$\text{Var}(s_{ij}) = d_k$$

**标准差 = √d_k。** 当 d_k = 64 时，标准差 = 8。这意味着点积的值可能达到 ±16 甚至更大。

为什么这有问题？因为 Softmax 对很大的值非常敏感：

```
Softmax([2, 1, 0]) = [0.67, 0.24, 0.09]  ← 还行，分布合理
Softmax([20, 10, 0]) = [1.00, 0.00, 0.00]  ← 梯度几乎为 0！
```

当输入值很大时，Softmax 的梯度趋近于零，导致训练困难（梯度消失）。

除以 √d_k 就把方差拉回 1，让 Softmax 工作在"舒适区"。

> 💡 记住：**缩放的目的是控制方差**，让 Softmax 的输入保持在一个梯度友好的范围。

**Step 3: Softmax 归一化**

对每一行做 Softmax：

$$A = \text{Softmax}(\hat{S}) \quad \text{按行归一化}$$

矩阵 A 的每一行和为 1，a_ij 表示 token i 对 token j 的注意力权重。

**Step 4: 加权求和**

$$\text{Output} = AV \quad \text{形状：[n, d_v]}$$

输出第 i 个 token 的表示 = 所有 token 的 V 按注意力权重加权求和：

$$\text{output}_i = \sum_{j=1}^{n} a_{ij} \cdot v_j$$

> 🎯 **一句话总结：每个 token 的新表示 = 它"看了"其他所有 token 之后，按关注度加权汇总的信息。**

---

### 3. 多头注意力（Multi-Head Attention）

#### 为什么要多头？

一个注意力头只能学一种"关注模式"。但语言中的关系是多样的：

- 有的头关注语法关系（主语→谓语）
- 有的头关注指代关系（"它"→前面的名词）
- 有的头关注位置关系（相邻的词）

**多头注意力 = 让模型同时从多个角度看问题。**

#### 数学定义

把 d_model 维度切成 h 个头，每个头的维度 d_k = d_model / h：

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \text{head}_2, ..., \text{head}_h) W_O$$

其中每个头：

$$\text{head}_i = \text{Attention}(QW_Q^i, KW_K^i, VW_V^i)$$

W_O 是输出的投影矩阵，形状 [d_model, d_model]。

#### 并行计算

所有头是**完全独立**的，可以并行计算。这就是为什么 GPU 上的注意力计算很高效。

> 🎨 类比：一个侦探查案，与其让一个人想破头，不如派出 8 个侦探，每个从不同角度调查，最后汇总结果。

#### GPT-2 的注意力配置

- d_model = 768
- 头数 h = 12
- 每个头维度 d_k = 768 / 12 = 64

---

### 4. 因果掩码（Causal Mask）

#### 为什么需要掩码？

在语言模型中，生成是**自回归**的——第 t 个 token 只能看到前 t-1 个 token，不能"偷看"未来的信息。

> 💡 **自回归（Autoregressive）**：生成过程是"一个接一个"的——每次基于已经生成的内容预测下一个。就像写作文，每写一个字都基于前面已经写的所有字。数学上表示为 P(x_t | x_1, x_2, ..., x_{t-1})。

> 📝 类比：写作文时，你写第 3 个字的时候，不可能看到第 5 个字——因为第 5 个字还不存在！

#### 掩码的数学实现

创建一个下三角矩阵 M：

```
M = [[0, -∞, -∞, -∞],
     [0,  0, -∞, -∞],
     [0,  0,  0, -∞],
     [0,  0,  0,  0]]
```

在 Softmax 之前加上掩码：

$$\hat{S}_{\text{masked}} = \hat{S} + M$$

因为 e^(-∞) = 0，所以被 -∞ 遮住的位置在 Softmax 后权重为 0，即"完全看不到"。

```python
# 用代码实现因果掩码
import torch
n = 4
mask = torch.triu(torch.ones(n, n), diagonal=1) * float('-inf')
# tensor([[0., -inf, -inf, -inf],
#         [0.,   0., -inf, -inf],
#         [0.,   0.,   0., -inf],
#         [0.,   0.,   0.,   0.]])
```

---

### 5. 注意力矩阵的分析

训练好模型后，我们可以可视化注意力矩阵，观察模型学到了什么模式。

#### 常见的注意力模式

| 模式 | 描述 | 示例 |
|------|------|------|
| 对角线 | 每个 token 关注自己 | 名词关注自身 |
| 前一行 | 关注前一个 token | 相邻词的关系 |
| 远程依赖 | 关注远处的 token | "它"→"猫"（指代消解） |
| 关注特殊 token | 所有 token 关注 [CLS] 或句首 | 全局信息聚合 |

> ❓ **暂停检查：** 你能想到为什么 GPT 用因果掩码（下三角），而 BERT 不需要吗？（提示：想想它们的训练目标有什么不同。）

---

### 6. KV Cache 的数学

#### 推理时的计算

在自回归生成中，假设已经生成了 t-1 个 token，现在要生成第 t 个。

每次生成一个新 token，理论上需要重新计算整个序列的注意力：

$$\text{Attention}(Q_{1:t}, K_{1:t}, V_{1:t})$$

但我们仔细看：**前 t-1 个 token 之间的注意力关系并没有改变！** 只有新 token 需要计算它对所有前面 token 的注意力。

#### KV Cache 的做法

缓存已经计算过的 K 和 V：

```
第1步：计算 q1, k1, v1 → 缓存 k1, v1 → output1 = Attn(q1, k1, v1)
第2步：计算 q2, k2, v2 → 缓存 k2, v2 → output2 = Attn(q2, [k1;k2], [v1;v2])
第3步：计算 q3, k3, v3 → 缓存 k3, v3 → output3 = Attn(q3, [k1;k2;k3], [v1;v2;v3])
...
```

> 💡 类比：考试写答案，你不需要每次都从头重新读题。你把题目记住了（缓存），每次只看新信息。

#### 复杂度对比

| 方法 | 第 t 步的计算量 | 总计算量（生成 n 个 token） |
|------|-----------------|---------------------------|
| 无缓存 | O(t · d) 每步 | O(n² · d) |
| 有 KV Cache | O(d) 每步 | O(n · d) |

**从 O(n²) 降到 O(n)，推理速度提升巨大！** 这就是为什么 KV Cache 是 LLM 推理的标配优化。

#### KV Cache 的显存开销

KV Cache 需要存储 2 × n_layers × n_tokens × d_model 个 float 值。

以 GPT-2 为例：12 层 × 768 维 × 2（K+V），生成 1024 个 token 的缓存大小：

$$2 \times 12 \times 1024 \times 768 \times 2 \text{ bytes (float16)} \approx 36 \text{ MB}$$

对于更大的模型（如 LLaMA-70B），KV Cache 会消耗数 GB 显存——这就是 **MQA（Multi-Query Attention）** 和 **GQA（Grouped-Query Attention）** 优化的动机。

---

## 🔢 完整公式推导（不跳步）

让我们用具体数字走一遍 Self-Attention 的完整计算。

### 设定

- 序列长度 n = 3（3 个 token）
- d_k = d_v = 4（为了演示，用小维度）
- 已计算好 Q、K、V

### 输入

```
Q = [[1.0, 0.5, -0.3, 0.8],    ← token 1 的 query
     [0.2, 1.0, 0.4, -0.1],     ← token 2 的 query
     [-0.5, 0.3, 1.2, 0.6]]     ← token 3 的 query

K = [[0.9, 0.3, 0.1, -0.5],    ← token 1 的 key
     [0.1, 0.8, -0.2, 0.7],     ← token 2 的 key
     [-0.3, 0.6, 1.0, 0.4]]     ← token 3 的 key

V = [[1.1, 0.4, -0.2, 0.5],    ← token 1 的 value
     [0.3, 0.9, 0.6, -0.3],     ← token 2 的 value
     [-0.4, 0.7, 1.1, 0.2]]     ← token 3 的 value
```

### Step 1: 计算注意力分数 S = QK^T

$$s_{11} = q_1 \cdot k_1 = 1.0 \times 0.9 + 0.5 \times 0.3 + (-0.3) \times 0.1 + 0.8 \times (-0.5) = 0.9 + 0.15 - 0.03 - 0.4 = 0.62$$

$$s_{12} = q_1 \cdot k_2 = 1.0 \times 0.1 + 0.5 \times 0.8 + (-0.3) \times (-0.2) + 0.8 \times 0.7 = 0.1 + 0.4 + 0.06 + 0.56 = 1.12$$

$$s_{13} = q_1 \cdot k_3 = 1.0 \times (-0.3) + 0.5 \times 0.6 + (-0.3) \times 1.0 + 0.8 \times 0.4 = -0.3 + 0.3 - 0.3 + 0.32 = 0.02$$

继续计算其余行...

$$S = \begin{bmatrix} 0.62 & 1.12 & 0.02 \\ -0.04 & 0.15 & 0.83 \\ -0.53 & 0.56 & 1.39 \end{bmatrix}$$

### Step 2: 缩放 Ŝ = S / √d_k

d_k = 4, √d_k = 2

$$\hat{S} = \frac{S}{2} = \begin{bmatrix} 0.31 & 0.56 & 0.01 \\ -0.02 & 0.075 & 0.415 \\ -0.265 & 0.28 & 0.695 \end{bmatrix}$$

### Step 3: 加因果掩码（可选，自回归模式时）

$$\hat{S}_{\text{masked}} = \hat{S} + \begin{bmatrix} 0 & -\infty & -\infty \\ 0 & 0 & -\infty \\ 0 & 0 & 0 \end{bmatrix} = \begin{bmatrix} 0.31 & -\infty & -\infty \\ -0.02 & 0.075 & -\infty \\ -0.265 & 0.28 & 0.695 \end{bmatrix}$$

### Step 4: Softmax（按行）

> 💡 **说明：** 这里以第 3 行为例进行 Softmax 计算。由于第 3 行对应的是序列中最后一个 token，在因果掩码下三角矩阵中，第 3 行的所有位置（1、2、3）都是可见的（没有被 -∞ 遮住），所以**加不加掩码，第 3 行的 Softmax 结果完全一样**。这就是我们选择第 3 行来演示的原因。

以第 3 行为例（不加掩码时）：

$$\text{row}_3 = [-0.265, 0.28, 0.695]$$

$$e^{-0.265} = 0.767, \quad e^{0.28} = 1.323, \quad e^{0.695} = 2.004$$

$$\text{sum} = 0.767 + 1.323 + 2.004 = 4.094$$

$$A_{\text{row3}} = [0.767/4.094, 1.323/4.094, 2.004/4.094] = [0.187, 0.323, 0.489]$$

**解读：** token 3 最关注 token 3 自己（48.9%），其次是 token 2（32.3%），最少关注 token 1（18.7%）。

### Step 5: 加权求和 Output = A × V

$$\text{output}_3 = 0.187 \times v_1 + 0.323 \times v_2 + 0.489 \times v_3$$

$$= 0.187 \times [1.1, 0.4, -0.2, 0.5] + 0.323 \times [0.3, 0.9, 0.6, -0.3] + 0.489 \times [-0.4, 0.7, 1.1, 0.2]$$

$$= [0.206, 0.075, -0.037, 0.094] + [0.097, 0.291, 0.194, -0.097] + [-0.196, 0.342, 0.538, 0.098]$$

$$= [0.107, 0.708, 0.695, 0.095]$$

**完成！** 这就是 token 3 经过 Self-Attention 后的新表示。它融入了整个序列的信息，而不仅仅是自己的嵌入。

> 🎯 **关键洞察：** 注意力的输出不是"替换"原始表示，而是创建了一个**融合了上下文信息的新表示**。在 Transformer 中，这个输出还会经过残差连接和 Layer Norm，与原始信息合并。

---

## 💻 代码验证

### 从零实现 Self-Attention

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ============================================
# 1. 从零实现 Scaled Dot-Product Attention
# ============================================

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Q: [batch, n_heads, seq_len, d_k]
    K: [batch, n_heads, seq_len, d_k]
    V: [batch, n_heads, seq_len, d_v]
    mask: [seq_len, seq_len] 或可广播的形状
    """
    d_k = Q.size(-1)
    
    # Step 1: 计算注意力分数
    scores = torch.matmul(Q, K.transpose(-2, -1))  # [batch, n_heads, seq_len, seq_len]
    
    # Step 2: 缩放
    scores = scores / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
    
    # Step 3: 加掩码（如果有）
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    
    # Step 4: Softmax
    attn_weights = F.softmax(scores, dim=-1)
    
    # Step 5: 加权求和
    output = torch.matmul(attn_weights, V)  # [batch, n_heads, seq_len, d_v]
    
    return output, attn_weights


# ============================================
# 2. 从零实现 Multi-Head Attention
# ============================================

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0, "d_model 必须能被 n_heads 整除"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # Q, K, V 的线性投影
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        
        # 输出投影
        self.W_o = nn.Linear(d_model, d_model)
    
    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.size()
        
        # 线性投影并分头
        Q = self.W_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # 计算注意力
        attn_output, attn_weights = scaled_dot_product_attention(Q, K, V, mask)
        
        # 合并多头
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        # 输出投影
        output = self.W_o(attn_output)
        
        return output, attn_weights


# ============================================
# 3. 可视化注意力热力图
# ============================================

def visualize_attention(attn_weights, tokens, title="注意力权重", save_path=None):
    """
    attn_weights: [n_heads, seq_len, seq_len] 或 [seq_len, seq_len]
    tokens: token 名称列表
    """
    if attn_weights.dim() == 4:
        attn_weights = attn_weights[0]  # 取 batch 中的第一个
    
    n_heads = attn_weights.size(0)
    fig, axes = plt.subplots(1, n_heads, figsize=(4 * n_heads, 4))
    if n_heads == 1:
        axes = [axes]
    
    for i, ax in enumerate(axes):
        sns.heatmap(attn_weights[i].detach().numpy(),
                    xticklabels=tokens, yticklabels=tokens,
                    cmap="YlOrRd", ax=ax, vmin=0, vmax=1,
                    annot=True, fmt='.2f', annot_kws={'size': 8})
        ax.set_title(f"Head {i+1}")
        ax.set_xlabel("Key (被看的)")
        ax.set_ylabel("Query (去看的)")
    
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"图片已保存到 {save_path}")
    plt.close()


# ============================================
# 4. 运行示例
# ============================================

if __name__ == "__main__":
    torch.manual_seed(42)
    
    # 参数设定（类似 GPT-2 mini）
    batch_size = 1
    seq_len = 6
    d_model = 24
    n_heads = 4
    d_k = d_model // n_heads  # = 6
    
    tokens = ["我", "喜欢", "吃", "苹果", "和", "香蕉"]
    
    # 随机输入
    x = torch.randn(batch_size, seq_len, d_model)
    
    # 创建因果掩码
    causal_mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(0)
    # causal_mask[i, j] = 1 表示可以看到，0 表示看不到
    
    # 初始化多头注意力
    mha = MultiHeadAttention(d_model, n_heads)
    
    # 前向传播
    output, attn_weights = mha(x, mask=causal_mask)
    
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print(f"注意力权重形状: {attn_weights.shape}")
    
    # 可视化
    visualize_attention(attn_weights, tokens, title="Multi-Head Attention 权重",
                       save_path="./images/attention_heatmap.png")
    
    # ---- 额外可视化：对比有/无因果掩码 ----
    _, attn_no_mask = mha(x, mask=None)
    _, attn_with_mask = mha(x, mask=causal_mask)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 取 head 0
    sns.heatmap(attn_no_mask[0, 0].detach().numpy(),
                xticklabels=tokens, yticklabels=tokens,
                cmap="YlOrRd", ax=ax1, annot=True, fmt='.2f')
    ax1.set_title("无因果掩码（双向注意力）")
    ax1.set_xlabel("Key")
    ax1.set_ylabel("Query")
    
    sns.heatmap(attn_with_mask[0, 0].detach().numpy(),
                xticklabels=tokens, yticklabels=tokens,
                cmap="YlOrRd", ax=ax2, annot=True, fmt='.2f')
    ax2.set_title("有因果掩码（单向注意力）")
    ax2.set_xlabel("Key")
    ax2.set_ylabel("Query")
    
    plt.suptitle("因果掩码对比", fontsize=14)
    plt.tight_layout()
    plt.savefig("./images/causal_mask_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("因果掩码对比图已保存")
    
    # ---- Softmax 温度对比 ----
    fig, axes = plt.subplots(1, 4, figsize=(16, 3))
    temperatures = [0.5, 1.0, 2.0, 5.0]
    test_scores = torch.tensor([2.0, 1.0, 0.5, 0.1])
    
    for ax, T in zip(axes, temperatures):
        probs = F.softmax(test_scores / T, dim=0)
        ax.bar(range(4), probs.numpy(), color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
        ax.set_title(f"T = {T}")
        ax.set_ylim(0, 1)
        ax.set_xticks(range(4))
        ax.set_xticklabels(['2.0', '1.0', '0.5', '0.1'])
    
    plt.suptitle("Softmax 温度参数效果", fontsize=14)
    plt.tight_layout()
    plt.savefig("./images/softmax_temperature.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Softmax 温度对比图已保存")
    
    print("\n✅ 所有可视化完成！")
```

![注意力热力图](./images/attention_heatmap.png)

*图：4 个注意力头的权重分布。颜色越深表示关注度越高。*

![因果掩码对比](./images/causal_mask_comparison.png)

*图：左边是双向注意力（BERT 风格），右边是因果注意力（GPT 风格）。注意右上角被遮住了。*

![Softmax温度对比](./images/softmax_temperature.png)

*图：不同温度下 Softmax 的输出分布。温度越低越集中，越高越均匀。*

---

## 🎯 LLM 关联

### GPT 系列的注意力配置

| 模型 | 层数 | d_model | 头数 | d_k | 参数量 |
|------|------|---------|------|-----|--------|
| GPT-2 Small | 12 | 768 | 12 | 64 | 117M |
| GPT-2 Medium | 24 | 1024 | 16 | 64 | 345M |
| GPT-2 Large | 36 | 1280 | 20 | 64 | 774M |
| GPT-2 XL | 48 | 1600 | 25 | 64 | 1.5B |
| GPT-3 | 96 | 12288 | 96 | 128 | 175B |

> 💡 注意一个有趣的点：GPT-2 的所有变体都使用 d_k = 64。这不是偶然——研究表明 64 是一个很好的平衡点，太小信息不够，太大计算量增加。

### 注意力参数量计算

每个 Transformer 层的注意力参数：

$$P_{\text{attn}} = 4 \times d_{\text{model}}^2$$

解释：W_Q、W_K、W_V、W_O 各是一个 d_model × d_model 的矩阵。

以 GPT-2 Small 为例：4 × 768² = 2,359,296 ≈ 2.4M 参数/层，12 层 ≈ 28.3M 仅注意力参数。

### Flash Attention 简介

标准注意力的瓶颈在于 **中间矩阵 A [n, n] 的显存占用**。当序列长度 n 很大时（如 n = 32K），这个矩阵就要占 32K × 32K × 4 bytes ≈ 4 GB！

**Flash Attention** 的核心思想：

1. **分块计算（Tiling）**：把 Q、K、V 切成小块，在 SRAM（快速缓存）中完成注意力计算
2. **不存储中间矩阵**：不需要把完整的 [n, n] 注意力矩阵写入 HBM（慢速显存）
3. **在线 Softmax**：通过数学技巧，在不知道全局 sum 的情况下正确计算 Softmax

> 🎨 类比：你要算全班 50 个人的平均分。标准方法是先算总分再除以 50（需要记住总分）。Flash Attention 的方法是每次看 5 个人，维护一个"运行中的平均值"，不需要记住所有人的分数。

**Flash Attention 的效果：**
- 显存：从 O(n²) 降到 O(n)
- 速度：实际提升 2-4 倍（减少了显存读写）
- 数学上**完全等价**（不是近似！）

---

## ❓ 思考题

### 题 1：缩放的直觉

如果 d_k = 256（而不是 64），不除以 √d_k 会发生什么？训练时会出现什么现象？

<details>
<summary>💡 提示</summary>

想想点积的方差会变成多少，Softmax 在大输入时的梯度会怎样。
</details>

### 题 2：多头的必要性

假设我们只用 1 个注意力头（h=1），d_k = d_model = 768。和用 12 个头（d_k = 64）相比，模型的表达能力有什么区别？

<details>
<summary>💡 提示</summary>

考虑一下"一个 768 维的注意力"和"12 个独立的 64 维注意力再拼接"，哪一种能学到更丰富的模式？
</details>

### 题 3：KV Cache 的极限

KV Cache 能无限增长吗？在实际系统中，如果生成 100K 个 token，会遇到什么问题？有哪些解决方案？

<details>
<summary>💡 提示</summary>

显存是有限的。想想 Paged Attention、Sliding Window Attention、MQA/GQA 这些技术。
</details>

### 题 4：注意力与卷积

有人说"Self-Attention 是一种全连接图上的消息传递"。CNN 的卷积也可以看作一种局部消息传递。请对比两者：
- 感受野（每层能看到多远）
- 计算复杂度（与序列长度的关系）
- 归纳偏置（inductive bias）

### 题 5：Flash Attention 的在线 Softmax

Flash Attention 的核心挑战之一：它需要在不知道全局 sum 的情况下计算 Softmax。试着用自己的话描述——**为什么 Flash Attention 不需要存储完整的注意力矩阵？** 它是怎么做到"边算边归一化"的？

<details>
<summary>💡 提示</summary>

想象你在批改试卷，但一次只能看 5 份。你可以维护两个变量：**当前见过的最高分 m** 和**当前所有分数的指数和 l**。每次看新的 5 份时，用新的最高分修正之前的累积和。这样你不需要一次性看完所有试卷，也能算出每个人的相对排名。Flash Attention 就是这个思路——分块处理，维护运行中的最大值和累积和，最终结果和一次性算完全等价。
</details>

---

## 📚 总结

| 概念 | 核心公式 | 一句话记忆 |
|------|---------|-----------|
| Softmax | e^zᵢ / Σe^zⱼ | 把分数变成概率 |
| 缩放点积注意力 | Softmax(QK^T/√d_k)V | 查询 × 键 → 权重 × 值 |
| 多头注意力 | Concat(head₁,...,headₕ)W_O | 多角度看问题 |
| 因果掩码 | S + 下三角(-∞) | 不偷看未来 |
| KV Cache | 缓存已计算的 K, V | 不重复计算 |

> 🌱 **下一章预告：** 我们将学习 Transformer 的另一个关键组件——前馈网络（FFN）和 Layer Normalization，以及它们背后的数学原理。

---

*"注意力不是你拥有的一切，但一切都需要注意力。"* — 差点就说这话的某位智者 😄
