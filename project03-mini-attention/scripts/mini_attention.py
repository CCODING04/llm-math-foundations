#!/usr/bin/env python3
"""
Mini Attention + 解码采样器
===========================
纯 NumPy 实现，不依赖 PyTorch / TensorFlow。

包含：
  1. Q/K/V 线性变换
  2. Scaled Dot-Product Attention
  3. Causal Mask（因果掩码）
  4. 多头注意力（Multi-Head Attention）
  5. Temperature / Top-K / Top-P 三种采样策略
  6. 不同采样参数的对比实验

运行方式：
  python mini_attention.py
"""

import numpy as np


# ══════════════════════════════════════════════════════════════
#  辅助函数
# ══════════════════════════════════════════════════════════════

def softmax(x, axis=-1):
    """数值稳定的 Softmax：先减最大值，防止溢出。"""
    x_max = np.max(x, axis=axis, keepdims=True)
    e_x = np.exp(x - x_max)
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


def create_causal_mask(seq_len):
    """
    创建因果掩码（下三角全 0，上三角全 -inf）。
    用于自回归模型：位置 i 只能看到位置 0..i。
    """
    return np.triu(np.full((seq_len, seq_len), -np.inf), k=1)


# ══════════════════════════════════════════════════════════════
#  步骤 1：Q/K/V 线性变换
# ══════════════════════════════════════════════════════════════

def linear_transform(X, W, b=None):
    """线性变换: Y = XW + b（b 可选）。"""
    out = X @ W
    if b is not None:
        out += b
    return out


def compute_qkv(X, W_Q, W_K, W_V, b_Q=None, b_K=None, b_V=None):
    """
    计算 Q、K、V。
    
    参数：
        X  : (seq_len, d_model) — 输入序列
        W_Q: (d_model, d_k)     — Query 投影矩阵
        W_K: (d_model, d_k)     — Key   投影矩阵
        W_V: (d_model, d_v)     — Value 投影矩阵
    返回：
        Q, K, V
    """
    Q = linear_transform(X, W_Q, b_Q)
    K = linear_transform(X, W_K, b_K)
    V = linear_transform(X, W_V, b_V)
    return Q, K, V


# ══════════════════════════════════════════════════════════════
#  步骤 2：Scaled Dot-Product Attention
# ══════════════════════════════════════════════════════════════

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Scaled Dot-Product Attention:
        Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V

    参数：
        Q   : (seq_len, d_k)
        K   : (seq_len, d_k)
        V   : (seq_len, d_v)
        mask: (seq_len, seq_len) 可选，需要屏蔽的位置填 -inf
    返回：
        output      : (seq_len, d_v)
        attn_weights: (seq_len, seq_len)
    """
    d_k = Q.shape[-1]

    # 1. QK^T → 注意力分数
    scores = Q @ K.T                           # (seq_len, seq_len)

    # 2. 缩放
    scores = scores / np.sqrt(d_k)

    # 3. 应用掩码（如果有）
    if mask is not None:
        scores = scores + mask

    # 4. Softmax → 注意力权重
    attn_weights = softmax(scores)              # (seq_len, seq_len)

    # 5. 加权求和
    output = attn_weights @ V                   # (seq_len, d_v)

    return output, attn_weights


# ══════════════════════════════════════════════════════════════
#  步骤 3：因果掩码（Causal Mask）—— 已在 create_causal_mask 实现
# ══════════════════════════════════════════════════════════════

# （逻辑集中在 create_causal_mask + scaled_dot_product_attention 的 mask 参数）


# ══════════════════════════════════════════════════════════════
#  步骤 4：多头注意力（Multi-Head Attention）
# ══════════════════════════════════════════════════════════════

class MultiHeadAttention:
    """
    纯 NumPy 实现的多头注意力。

    流程：分头 Q/K/V → 独立 Attention → 拼接 → 输出投影
    """

    def __init__(self, d_model, num_heads, seed=42):
        """
        参数：
            d_model  : 模型维度（必须能被 num_heads 整除）
            num_heads: 注意力头数
            seed     : 随机种子
        """
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_v = d_model // num_heads

        rng = np.random.default_rng(seed)

        # 每个头独立的 Q/K/V 投影矩阵
        self.W_Qs = [rng.standard_normal((d_model, self.d_k)) * 0.1
                     for _ in range(num_heads)]
        self.W_Ks = [rng.standard_normal((d_model, self.d_k)) * 0.1
                     for _ in range(num_heads)]
        self.W_Vs = [rng.standard_normal((d_model, self.d_v)) * 0.1
                     for _ in range(num_heads)]

        # 输出投影矩阵
        self.W_O = rng.standard_normal((num_heads * self.d_v, d_model)) * 0.1

    def forward(self, X, mask=None):
        """
        前向传播。

        参数：
            X   : (seq_len, d_model)
            mask: (seq_len, seq_len) 可选
        返回：
            output: (seq_len, d_model)
        """
        head_outputs = []

        for h in range(self.num_heads):
            Q_h = X @ self.W_Qs[h]              # (seq_len, d_k)
            K_h = X @ self.W_Ks[h]
            V_h = X @ self.W_Vs[h]

            out_h, _ = scaled_dot_product_attention(
                Q_h, K_h, V_h, mask=mask
            )                                    # (seq_len, d_v)
            head_outputs.append(out_h)

        # 拼接: (seq_len, num_heads * d_v)
        concat = np.concatenate(head_outputs, axis=-1)

        # 输出投影: (seq_len, d_model)
        output = concat @ self.W_O
        return output


# ══════════════════════════════════════════════════════════════
#  步骤 5：采样策略
# ══════════════════════════════════════════════════════════════

def temperature_sampling(logits, temperature=1.0):
    """
    Temperature 采样。

    参数：
        logits     : (vocab_size,) 原始分数
        temperature: 温度系数，T>1 更随机，T<1 更确定
    返回：
        采样的 token 索引
    """
    if temperature <= 0:
        # T→0 退化为贪心
        return np.argmax(logits)

    scaled_logits = logits / temperature
    probs = softmax(scaled_logits)
    return np.random.choice(len(probs), p=probs)


def top_k_sampling(logits, k=10):
    """
    Top-K 采样：只保留概率最大的 K 个 token。

    参数：
        logits: (vocab_size,)
        k     : 保留的 token 数量
    返回：
        采样的 token 索引
    """
    if k <= 0:
        k = len(logits)
    k = min(k, len(logits))

    # 找 top-k 索引
    top_k_indices = np.argsort(logits)[-k:]

    # 过滤：非 top-k 设为 -inf
    filtered_logits = np.full_like(logits, -np.inf)
    filtered_logits[top_k_indices] = logits[top_k_indices]

    probs = softmax(filtered_logits)
    return np.random.choice(len(probs), p=probs)


def top_p_sampling(logits, p=0.9):
    """
    Top-P（核采样）：按概率从大到小排列，累积达到 p 时截断。

    参数：
        logits: (vocab_size,)
        p     : 累积概率阈值
    返回：
        采样的 token 索引
    """
    if p >= 1.0:
        # 不截断，等价于普通随机采样
        probs = softmax(logits)
        return np.random.choice(len(probs), p=probs)

    # 按概率降序排列
    sorted_indices = np.argsort(logits)[::-1]
    sorted_logits = logits[sorted_indices]
    sorted_probs = softmax(sorted_logits)
    cumulative_probs = np.cumsum(sorted_probs)

    # 找到累积概率超过 p 的位置
    cutoff_idx = np.searchsorted(cumulative_probs, p) + 1
    cutoff_idx = min(cutoff_idx, len(logits))

    # 只保留前 cutoff_idx 个
    filtered_logits = np.full_like(logits, -np.inf)
    filtered_logits[sorted_indices[:cutoff_idx]] = logits[sorted_indices[:cutoff_idx]]

    probs = softmax(filtered_logits)
    return np.random.choice(len(probs), p=probs)


# ══════════════════════════════════════════════════════════════
#  步骤 6：对比实验
# ══════════════════════════════════════════════════════════════

def run_sampling_comparison():
    """对比不同采样参数的输出分布。"""
    print("=" * 60)
    print("  采样策略对比实验")
    print("=" * 60)

    # 模拟 logits（假设 10 个 token）
    logits = np.array([3.0, 2.5, 2.0, 1.5, 1.0, 0.5, 0.2, 0.1, 0.05, 0.01])
    vocab = [f"t{i}" for i in range(10)]

    # ─── Temperature 对比 ─────────────────────────────
    print("\n--- Temperature 对比 ---")
    print(f"原始 logits: {logits}")
    print()

    for T in [0.1, 0.5, 1.0, 2.0, 5.0]:
        probs = softmax(logits / T)
        print(f"T={T:<4} → {probs.round(4)}")

    # 多次采样，观察分布
    np.random.seed(0)
    n_samples = 1000
    print(f"\n采样 {n_samples} 次的频率分布:")
    for T in [0.5, 1.0, 2.0]:
        counts = np.zeros(len(logits), dtype=int)
        for _ in range(n_samples):
            idx = temperature_sampling(logits, temperature=T)
            counts[idx] += 1
        freq = counts / n_samples
        print(f"T={T:<4} → {freq.round(3)}")

    # ─── Top-K 对比 ──────────────────────────────────
    print("\n--- Top-K 对比 ---")

    np.random.seed(0)
    for k in [1, 3, 5, 10]:
        counts = np.zeros(len(logits), dtype=int)
        for _ in range(n_samples):
            idx = top_k_sampling(logits, k=k)
            counts[idx] += 1
        freq = counts / n_samples
        print(f"K={k:<3} → {freq.round(3)}")

    # ─── Top-P 对比 ──────────────────────────────────
    print("\n--- Top-P 对比 ---")

    np.random.seed(0)
    for p_val in [0.3, 0.5, 0.9, 1.0]:
        counts = np.zeros(len(logits), dtype=int)
        for _ in range(n_samples):
            idx = top_p_sampling(logits, p=p_val)
            counts[idx] += 1
        freq = counts / n_samples
        print(f"P={p_val:<4} → {freq.round(3)}")

    # ─── 组合策略：Temperature + Top-K / Top-P ─────
    print("\n--- 组合策略 (Temperature + Top-P) ---")

    np.random.seed(0)
    configs = [
        {"T": 0.7, "p": 0.9, "label": "T=0.7, P=0.9 (常见配置)"},
        {"T": 1.0, "p": 0.95, "label": "T=1.0, P=0.95 (平衡)"},
        {"T": 1.5, "p": 0.95, "label": "T=1.5, P=0.95 (创意)"},
    ]

    for cfg in configs:
        counts = np.zeros(len(logits), dtype=int)
        for _ in range(n_samples):
            # 先用 Top-P 过滤，再用 Temperature 缩放
            p_val = cfg["p"]
            T = cfg["T"]

            # Top-P 过滤
            sorted_indices = np.argsort(logits)[::-1]
            sorted_logits = logits[sorted_indices]
            sorted_probs = softmax(sorted_logits)
            cumulative = np.cumsum(sorted_probs)
            cutoff = np.searchsorted(cumulative, p_val) + 1
            cutoff = min(cutoff, len(logits))

            filtered = np.full_like(logits, -np.inf)
            filtered[sorted_indices[:cutoff]] = logits[sorted_indices[:cutoff]]

            # Temperature 缩放
            probs = softmax(filtered / T)
            idx = np.random.choice(len(probs), p=probs)
            counts[idx] += 1

        freq = counts / n_samples
        print(f"{cfg['label']}")
        print(f"         → {freq.round(3)}")


# ══════════════════════════════════════════════════════════════
#  主程序
# ══════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  🚀 Mini Attention + 解码采样器")
    print("  纯 NumPy 实现 · 不依赖 PyTorch")
    print("=" * 60)

    # ─── 参数设置 ────────────────────────────────────
    seq_len = 5       # 序列长度
    d_model = 8       # 模型维度
    num_heads = 2     # 注意力头数
    d_k = d_model // num_heads   # 每个头的 Q/K 维度 = 4
    d_v = d_model // num_heads   # 每个头的 V 维度 = 4

    np.random.seed(42)

    # ─── 步骤 1：Q/K/V 线性变换 ──────────────────────
    print("\n" + "─" * 60)
    print("  步骤 1：Q/K/V 线性变换")
    print("─" * 60)

    X = np.random.randn(seq_len, d_model)
    W_Q = np.random.randn(d_model, d_k) * 0.1
    W_K = np.random.randn(d_model, d_k) * 0.1
    W_V = np.random.randn(d_model, d_v) * 0.1

    Q, K, V = compute_qkv(X, W_Q, W_K, W_V)

    print(f"  输入 X   : shape = {X.shape}  (seq_len={seq_len}, d_model={d_model})")
    print(f"  权重 W_Q : shape = {W_Q.shape}  (d_model={d_model}, d_k={d_k})")
    print(f"  Query Q  : shape = {Q.shape}")
    print(f"  Key K    : shape = {K.shape}")
    print(f"  Value V  : shape = {V.shape}")

    # ─── 步骤 2：Scaled Dot-Product Attention ──────
    print("\n" + "─" * 60)
    print("  步骤 2：Scaled Dot-Product Attention")
    print("─" * 60)

    output_no_mask, attn_no_mask = scaled_dot_product_attention(Q, K, V)

    print(f"  注意力权重 shape : {attn_no_mask.shape}")
    print(f"  输出 shape       : {output_no_mask.shape}")
    print(f"\n  无掩码注意力权重:")
    for i in range(seq_len):
        row = "  ".join(f"{v:.3f}" for v in attn_no_mask[i])
        print(f"    位置 {i}: [{row}]")

    # ─── 步骤 3：因果掩码 ──────────────────────────
    print("\n" + "─" * 60)
    print("  步骤 3：因果掩码 (Causal Mask)")
    print("─" * 60)

    causal_mask = create_causal_mask(seq_len)
    output_masked, attn_masked = scaled_dot_product_attention(
        Q, K, V, mask=causal_mask
    )

    print(f"  因果掩码 (0=可见, -inf=屏蔽):")
    for i in range(seq_len):
        row = "  ".join(
            f"  0  " if causal_mask[i, j] == 0 else "-inf"
            for j in range(seq_len)
        )
        print(f"    位置 {i}: [{row}]")

    print(f"\n  掩码后注意力权重（上三角应为 0）:")
    for i in range(seq_len):
        row = "  ".join(f"{v:.3f}" for v in attn_masked[i])
        print(f"    位置 {i}: [{row}]")

    # 验证：每个位置的权重只分配给当前及之前位置
    print("\n  ✅ 验证：每个位置只能看到自己和之前的位置")
    for i in range(seq_len):
        visible = attn_masked[i, :i+1]
        total = visible.sum()
        print(f"    位置 {i} → 可见位置 0..{i} 的权重和 = {total:.6f}")

    # ─── 步骤 4：多头注意力 ─────────────────────────
    print("\n" + "─" * 60)
    print("  步骤 4：多头注意力 (Multi-Head Attention)")
    print("─" * 60)

    mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads, seed=42)
    mha_output = mha.forward(X, mask=causal_mask)

    print(f"  输入 shape          : {X.shape}")
    print(f"  注意力头数          : {num_heads}")
    print(f"  每个头的 d_k / d_v  : {d_k}")
    print(f"  多头注意力输出 shape: {mha_output.shape}")
    print(f"  ✅ 输入输出 shape 一致: {X.shape == mha_output.shape}")

    # 对比：有掩码 vs 无掩码
    mha_no_mask = mha.forward(X, mask=None)
    diff = np.abs(mha_output - mha_no_mask).mean()
    print(f"\n  有掩码 vs 无掩码输出差异 (MAE): {diff:.6f}")

    # ─── 步骤 5 & 6：采样策略对比 ────────────────────
    print("\n" + "─" * 60)
    print("  步骤 5 & 6：采样策略")
    print("─" * 60)

    run_sampling_comparison()

    # ─── 总结 ────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  ✅ 全部完成！")
    print("=" * 60)
    print("""
  你刚刚手写了：
    ✓ Q/K/V 线性变换
    ✓ Scaled Dot-Product Attention（含缩放因子 √d_k）
    ✓ Causal Mask（因果掩码，禁止偷看未来）
    ✓ Multi-Head Attention（多头注意力，分头→拼接→投影）
    ✓ Temperature / Top-K / Top-P 三种采样策略
    ✓ 不同采样参数的对比实验

  这些就是 Transformer 和 LLM 解码的核心组件！
    """)


if __name__ == "__main__":
    main()
