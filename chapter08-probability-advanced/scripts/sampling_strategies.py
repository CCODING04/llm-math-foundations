"""完整对比不同采样策略生成的 token 序列"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Heiti SC', 'Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

def softmax(logits, T=1.0):
    scaled = logits / T
    exp_l = np.exp(scaled - np.max(scaled))
    return exp_l / exp_l.sum()

def greedy_search(probs):
    return np.argmax(probs)

def random_sample(probs):
    return np.random.choice(len(probs), p=probs)

def top_k_sample(probs, k):
    sorted_idx = np.argsort(probs)[::-1]
    top_k_idx = sorted_idx[:k]
    top_k_probs = probs[top_k_idx]
    top_k_probs = top_k_probs / top_k_probs.sum()
    chosen = np.random.choice(top_k_idx, p=top_k_probs)
    return chosen

def top_p_sample(probs, p):
    sorted_idx = np.argsort(probs)[::-1]
    sorted_probs = probs[sorted_idx]
    cumsum = np.cumsum(sorted_probs)
    cutoff = np.searchsorted(cumsum, p) + 1
    nucleus_idx = sorted_idx[:cutoff]
    nucleus_probs = probs[nucleus_idx]
    nucleus_probs = nucleus_probs / nucleus_probs.sum()
    chosen = np.random.choice(nucleus_idx, p=nucleus_probs)
    return chosen

# 模拟一个简化的 vocab（8 个 token）
np.random.seed(42)
vocab = ["我", "想", "吃", "去", "学习", "看", "玩", "睡觉"]

# 模拟 10 步生成的 logits
print("=" * 60)
print("不同采样策略的生成结果对比")
print("=" * 60)

# 固定 logits 序列（模拟确定性模型输出）
all_logits = np.random.randn(10, 8) * 2 + np.array([3, 2, 1, 0.5, 0.3, 0.1, -0.5, -1])

strategies = {
    "贪心搜索": lambda p: greedy_search(p),
    "随机采样 (T=1)": lambda p: random_sample(p),
    "随机采样 (T=0.5)": lambda p: random_sample(softmax(
        np.log(p / (p + 1e-10) + 1e-10), T=0.5)),  # 近似恢复 logits
    "Top-K (K=3)": lambda p: top_k_sample(p, 3),
    "Top-P (P=0.9)": lambda p: top_p_sample(p, 0.9),
}

results = {}
for name, strategy in strategies.items():
    tokens = []
    for t in range(10):
        logits = all_logits[t]
        probs = softmax(logits)
        idx = strategy(probs)
        tokens.append(vocab[idx])
    results[name] = "".join(tokens)
    print(f"\n{name}: {''.join(tokens)}")

# 可视化：每一步的概率分布和选择
fig, axes = plt.subplots(2, 5, figsize=(20, 8))
fig.suptitle("贪心搜索 vs Top-P 采样：每步的选择", fontsize=16, fontweight='bold')

for step in range(10):
    row, col = step // 5, step % 5
    logits = all_logits[step]
    probs = softmax(logits)

    axes[row, col].bar(vocab, probs, color='lightsteelblue')
    # 标记贪心选择
    greedy_idx = np.argmax(probs)
    axes[row, col].bar(vocab[greedy_idx], probs[greedy_idx], color='coral', alpha=0.7)
    axes[row, col].set_title(f"Step {step+1}", fontsize=11)
    axes[row, col].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig("./images/sampling_steps.png", dpi=150, bbox_inches='tight')
plt.close()
print("\n✅ sampling_steps.png saved")
