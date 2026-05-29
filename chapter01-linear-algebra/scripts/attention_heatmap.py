"""
Transformer 注意力权重热力图
展示自注意力机制中 token 之间的注意力分布
"""
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['PingFang SC', 'Heiti SC', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False
import numpy as np

# 模拟一个句子的注意力权重
tokens = ['猫', '坐在', '柔软的', '垫子', '上']

# 模拟注意力权重矩阵（手工设计，展示语义关联）
attention = np.array([
    [0.60, 0.15, 0.05, 0.15, 0.05],  # 猫 → 猫(自注意高), 坐在(主谓), 垫子(语义)
    [0.20, 0.45, 0.05, 0.20, 0.10],  # 坐在 → 猫(主语), 垫子(宾语)
    [0.05, 0.05, 0.30, 0.55, 0.05],  # 柔软的 → 垫子(修饰)
    [0.25, 0.15, 0.25, 0.30, 0.05],  # 垫子 → 猫(语义), 坐在, 柔软的(被修饰)
    [0.10, 0.30, 0.05, 0.15, 0.40],  # 上 → 坐在(搭配), 自身
])

fig, ax = plt.subplots(figsize=(8, 6))

im = ax.imshow(attention, cmap='YlOrRd', aspect='auto', vmin=0, vmax=0.7)

# 标注数值
for i in range(len(tokens)):
    for j in range(len(tokens)):
        color = 'white' if attention[i, j] > 0.35 else 'black'
        ax.text(j, i, f'{attention[i, j]:.2f}', ha='center', va='center',
                fontsize=11, color=color, fontweight='bold')

ax.set_xticks(range(len(tokens)))
ax.set_yticks(range(len(tokens)))
ax.set_xticklabels(tokens, fontsize=12)
ax.set_yticklabels(tokens, fontsize=12)
ax.set_xlabel('Key（被关注的 token）', fontsize=12, fontweight='bold')
ax.set_ylabel('Query（发出关注的 token）', fontsize=12, fontweight='bold')
ax.set_title('自注意力权重矩阵示例\n"猫 坐在 柔软的 垫子 上"', fontsize=14, fontweight='bold')

plt.colorbar(im, ax=ax, label='注意力权重', shrink=0.8)
plt.tight_layout()
plt.savefig('/tmp/llm-math-foundations/chapter01-linear-algebra/images/attention_heatmap.png', dpi=150, bbox_inches='tight')
print("✅ 已保存: images/attention_heatmap.png")
