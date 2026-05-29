"""
第七章：注意力机制 — 可视化脚本
生成注意力热力图、因果掩码对比、Softmax温度对比
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# 输出目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
IMG_DIR = os.path.join(PROJECT_DIR, "images")
os.makedirs(IMG_DIR, exist_ok=True)


# ============================================
# 1. 从零实现 Scaled Dot-Product Attention
# ============================================

def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1))
    scores = scores / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    attn_weights = F.softmax(scores, dim=-1)
    output = torch.matmul(attn_weights, V)
    return output, attn_weights


# ============================================
# 2. 从零实现 Multi-Head Attention
# ============================================

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.size()
        Q = self.W_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        attn_output, attn_weights = scaled_dot_product_attention(Q, K, V, mask)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.W_o(attn_output)
        return output, attn_weights


def main():
    torch.manual_seed(42)

    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    batch_size = 1
    seq_len = 6
    d_model = 24
    n_heads = 4

    tokens = ["我", "喜欢", "吃", "苹果", "和", "香蕉"]

    x = torch.randn(batch_size, seq_len, d_model)

    causal_mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(0)

    mha = MultiHeadAttention(d_model, n_heads)
    output, attn_weights = mha(x, mask=causal_mask)

    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print(f"注意力权重形状: {attn_weights.shape}")

    # ---- 图1: 多头注意力热力图 ----
    if attn_weights.dim() == 4:
        weights = attn_weights[0]  # [n_heads, seq_len, seq_len]
    else:
        weights = attn_weights

    fig, axes = plt.subplots(1, n_heads, figsize=(4 * n_heads, 4))
    if n_heads == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        sns.heatmap(weights[i].detach().numpy(),
                    xticklabels=tokens, yticklabels=tokens,
                    cmap="YlOrRd", ax=ax, vmin=0, vmax=1,
                    annot=True, fmt='.2f', annot_kws={'size': 9})
        ax.set_title(f"Head {i+1}", fontsize=12)
        ax.set_xlabel("Key (被看的)", fontsize=10)
        ax.set_ylabel("Query (去看的)", fontsize=10)

    plt.suptitle("Multi-Head Attention 权重分布", fontsize=14, fontweight='bold')
    plt.tight_layout()
    path1 = os.path.join(IMG_DIR, "attention_heatmap.png")
    plt.savefig(path1, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ 图片已保存: {path1}")

    # ---- 图2: 因果掩码对比 ----
    _, attn_no_mask = mha(x, mask=None)
    _, attn_with_mask = mha(x, mask=causal_mask)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    sns.heatmap(attn_no_mask[0, 0].detach().numpy(),
                xticklabels=tokens, yticklabels=tokens,
                cmap="YlOrRd", ax=ax1, annot=True, fmt='.2f')
    ax1.set_title("无因果掩码（双向注意力 / BERT风格）", fontsize=12)
    ax1.set_xlabel("Key", fontsize=10)
    ax1.set_ylabel("Query", fontsize=10)

    sns.heatmap(attn_with_mask[0, 0].detach().numpy(),
                xticklabels=tokens, yticklabels=tokens,
                cmap="YlOrRd", ax=ax2, annot=True, fmt='.2f')
    ax2.set_title("有因果掩码（单向注意力 / GPT风格）", fontsize=12)
    ax2.set_xlabel("Key", fontsize=10)
    ax2.set_ylabel("Query", fontsize=10)

    plt.suptitle("因果掩码对比", fontsize=14, fontweight='bold')
    plt.tight_layout()
    path2 = os.path.join(IMG_DIR, "causal_mask_comparison.png")
    plt.savefig(path2, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ 图片已保存: {path2}")

    # ---- 图3: Softmax 温度对比 ----
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.5))
    temperatures = [0.5, 1.0, 2.0, 5.0]
    test_scores = torch.tensor([2.0, 1.0, 0.5, 0.1])
    labels = ['2.0', '1.0', '0.5', '0.1']
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']

    for ax, T in zip(axes, temperatures):
        probs = F.softmax(test_scores / T, dim=0)
        bars = ax.bar(range(4), probs.numpy(), color=colors, edgecolor='white', linewidth=1.5)
        ax.set_title(f"T = {T}", fontsize=12, fontweight='bold')
        ax.set_ylim(0, 1.05)
        ax.set_xticks(range(4))
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_ylabel("概率", fontsize=10)
        # 在柱子上标数值
        for bar, p in zip(bars, probs):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{p:.3f}', ha='center', va='bottom', fontsize=9)

    plt.suptitle("Softmax 温度参数效果对比", fontsize=14, fontweight='bold')
    plt.tight_layout()
    path3 = os.path.join(IMG_DIR, "softmax_temperature.png")
    plt.savefig(path3, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ 图片已保存: {path3}")

    # ---- 图4: KV Cache 加速示意图 ----
    fig, ax = plt.subplots(figsize=(8, 5))
    seq_lengths = np.arange(1, 501)
    no_cache = seq_lengths ** 2
    with_cache = seq_lengths * 1  # O(n) vs O(n²)
    
    ax.plot(seq_lengths, no_cache / no_cache.max(), label='无 KV Cache: O(n²)', 
            color='#FF6B6B', linewidth=2.5)
    ax.plot(seq_lengths, with_cache / with_cache.max(), label='有 KV Cache: O(n)', 
            color='#4ECDC4', linewidth=2.5)
    ax.fill_between(seq_lengths, no_cache / no_cache.max(), alpha=0.1, color='#FF6B6B')
    ax.fill_between(seq_lengths, with_cache / with_cache.max(), alpha=0.1, color='#4ECDC4')
    ax.set_xlabel("序列长度 (tokens)", fontsize=12)
    ax.set_ylabel("相对计算量", fontsize=12)
    ax.set_title("KV Cache 对推理计算量的影响", fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path4 = os.path.join(IMG_DIR, "kv_cache_comparison.png")
    plt.savefig(path4, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ 图片已保存: {path4}")

    print(f"\n🎉 所有可视化完成！图片保存在: {IMG_DIR}")


if __name__ == "__main__":
    main()
