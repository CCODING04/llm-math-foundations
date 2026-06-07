"""对比 Top-K 和 Top-P 采样的截断效果"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Heiti SC', 'Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

def softmax(logits):
    exp_l = np.exp(logits - np.max(logits))
    return exp_l / exp_l.sum()

def top_k_filter(probs, k):
    """Top-K 过滤"""
    sorted_indices = np.argsort(probs)[::-1]
    filtered = np.zeros_like(probs)
    filtered[sorted_indices[:k]] = probs[sorted_indices[:k]]
    return filtered / filtered.sum()

def top_p_filter(probs, p):
    """Top-P（核采样）过滤"""
    sorted_indices = np.argsort(probs)[::-1]
    sorted_probs = probs[sorted_indices]
    cumsum = np.cumsum(sorted_probs)
    # 找到累积概率 >= p 的位置
    cutoff = np.searchsorted(cumsum, p) + 1
    filtered = np.zeros_like(probs)
    filtered[sorted_indices[:cutoff]] = probs[sorted_indices[:cutoff]]
    return filtered / filtered.sum()

# 模拟两种分布：尖锐 vs 平坦
logits_sharp = np.array([5.0, 3.0, 1.0, 0.5, 0.3, 0.1, 0.05, 0.01])
logits_flat = np.array([1.5, 1.3, 1.1, 0.9, 0.7, 0.5, 0.3, 0.1])
tokens = [f"w{i}" for i in range(8)]

probs_sharp = softmax(logits_sharp)
probs_flat = softmax(logits_flat)

fig, axes = plt.subplots(2, 4, figsize=(20, 10))

for row, (probs, label) in enumerate([(probs_sharp, "尖锐分布"), (probs_flat, "平坦分布")]):
    # 原始分布
    axes[row, 0].bar(tokens, probs, color='steelblue')
    axes[row, 0].set_title(f"{label}\n原始分布")

    # Top-K = 3
    topk_probs = top_k_filter(probs, 3)
    axes[row, 1].bar(tokens, topk_probs, color='coral')
    axes[row, 1].set_title(f"Top-K (K=3)")

    # Top-P = 0.9
    topp_probs = top_p_filter(probs, 0.9)
    axes[row, 2].bar(tokens, topp_probs, color='seagreen')
    axes[row, 2].set_title(f"Top-P (P=0.9)")

    # 累积概率曲线
    sorted_p = np.sort(probs)[::-1]
    cumsum = np.cumsum(sorted_p)
    axes[row, 3].plot(range(1, 9), cumsum, 'o-', color='purple')
    axes[row, 3].axhline(y=0.9, color='red', linestyle='--', label='P=0.9')
    axes[row, 3].set_title("累积概率曲线")
    axes[row, 3].legend()
    axes[row, 3].set_xlabel("Token 排名")
    axes[row, 3].set_ylabel("累积概率")

plt.tight_layout()
plt.savefig("./images/topk_topp_comparison.png", dpi=150, bbox_inches='tight')
plt.close()
print("✅ topk_topp_comparison.png saved")
