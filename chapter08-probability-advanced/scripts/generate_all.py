#!/usr/bin/env python3
"""生成第八章所有可视化图片"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

# 中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'PingFang SC', 'Heiti SC']
plt.rcParams['axes.unicode_minus'] = False

# 确保图片目录存在
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(script_dir, '..', 'images')
os.makedirs(images_dir, exist_ok=True)

# ========== 1. 温度系数效果 ==========
logits = np.array([2.0, 1.5, 1.0, 0.5, 0.1, -0.5, -1.0, -1.5])

temperatures = [0.5, 1.0, 2.0]
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
tokens = [f'Token{i}' for i in range(8)]

for ax, T in zip(axes, temperatures):
    probs = np.exp(logits / T) / np.sum(np.exp(logits / T))
    ax.bar(tokens, probs, color=plt.cm.Blues(np.linspace(0.3, 0.9, 8)))
    ax.set_title(f'Temperature = {T}', fontsize=13)
    ax.set_ylabel('概率')
    ax.set_ylim(0, 0.7)
    for i, p in enumerate(probs):
        ax.text(i, p + 0.01, f'{p:.3f}', ha='center', fontsize=8)

plt.suptitle('温度系数对概率分布的影响', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'temperature_effect.png'), dpi=150, bbox_inches='tight')
plt.close()
print("✅ temperature_effect.png")

# ========== 2. Top-K vs Top-P 对比 ==========
logits_sample = np.array([3.0, 2.0, 1.0, 0.5, 0.2, 0.1, -0.5, -1.0, -1.5, -2.0])
tokens10 = [f'T{i}' for i in range(10)]
probs = np.exp(logits_sample) / np.sum(np.exp(logits_sample))

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

# 原始分布
axes[0].bar(tokens10, probs, color='steelblue')
axes[0].set_title('原始 Softmax 分布', fontsize=13)
axes[0].set_ylabel('概率')

# Top-K (K=3)
top_k = 3
top_k_probs = np.zeros_like(probs)
top_indices = np.argsort(probs)[-top_k:]
top_k_probs[top_indices] = probs[top_indices]
top_k_probs = top_k_probs / top_k_probs.sum()
colors_k = ['tomato' if i in top_indices else 'lightgray' for i in range(10)]
axes[1].bar(tokens10, top_k_probs, color=colors_k)
axes[1].set_title(f'Top-K 采样 (K={top_k})', fontsize=13)

# Top-P (P=0.8)
sorted_idx = np.argsort(probs)[::-1]
cumsum = np.cumsum(probs[sorted_idx])
cutoff = np.searchsorted(cumsum, 0.8) + 1
selected = sorted_idx[:cutoff]
top_p_probs = np.zeros_like(probs)
top_p_probs[selected] = probs[selected]
top_p_probs = top_p_probs / top_p_probs.sum()
colors_p = ['coral' if i in selected else 'lightgray' for i in range(10)]
axes[2].bar(tokens10, top_p_probs, color=colors_p)
axes[2].set_title(f'Top-P 核采样 (P=0.8)', fontsize=13)

plt.suptitle('Top-K vs Top-P 采样策略对比', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'topk_topp_comparison.png'), dpi=150, bbox_inches='tight')
plt.close()
print("✅ topk_topp_comparison.png")

# ========== 3. 采样步骤可视化 ==========
np.random.seed(42)
fig, axes = plt.subplots(1, 4, figsize=(18, 4))

vocab = ['猫', '狗', '鸟', '鱼', '虫', '花']
probs = np.array([0.35, 0.25, 0.15, 0.10, 0.08, 0.07])

# Step 1: 原始分布
axes[0].barh(vocab, probs, color='steelblue')
axes[0].set_title('Step 1: 模型输出概率', fontsize=12)
axes[0].set_xlabel('概率')

# Step 2: 温度调整 (T=0.7)
T = 0.7
adjusted = np.exp(np.log(probs + 1e-10) / T)
adjusted = adjusted / adjusted.sum()
axes[1].barh(vocab, adjusted, color='coral')
axes[1].set_title(f'Step 2: 温度调整 (T={T})', fontsize=12)
axes[1].set_xlabel('概率')

# Step 3: Top-K (K=3)
top3 = np.argsort(adjusted)[-3:]
filtered = np.zeros_like(adjusted)
filtered[top3] = adjusted[top3]
filtered = filtered / filtered.sum()
colors3 = ['tomato' if i in top3 else 'lightgray' for i in range(6)]
axes[2].barh(vocab, filtered, color=colors3)
axes[2].set_title('Step 3: Top-K 过滤 (K=3)', fontsize=12)
axes[2].set_xlabel('概率')

# Step 4: 采样结果
sampled = np.random.choice(vocab, p=filtered, size=1000)
counts = {v: (sampled == v).sum() for v in vocab}
axes[3].barh(vocab, [counts[v]/1000 for v in vocab], color='seagreen')
axes[3].set_title('Step 4: 采样结果 (1000次)', fontsize=12)
axes[3].set_xlabel('频率')

plt.suptitle('LLM 文本生成的采样步骤', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'sampling_steps.png'), dpi=150, bbox_inches='tight')
plt.close()
print("✅ sampling_steps.png")

print("\n🎉 所有图片生成完毕！")
