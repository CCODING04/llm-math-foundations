"""可视化温度系数对概率分布的影响"""
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'PingFang SC', 'Heiti TC']
plt.rcParams['axes.unicode_minus'] = False

# 模拟一个 vocab 的 logits（比如 10 个 token）
logits = np.array([4.0, 2.5, 2.0, 1.5, 1.0, 0.5, 0.3, 0.1, -0.5, -1.0])
tokens = [f"w{i}" for i in range(10)]

def softmax_with_temperature(logits, T):
    """带温度的 Softmax"""
    scaled = logits / T
    exp_scaled = np.exp(scaled - np.max(scaled))  # 数值稳定性
    return exp_scaled / exp_scaled.sum()

temperatures = [0.1, 0.5, 1.0, 2.0, 5.0]

fig, axes = plt.subplots(1, 5, figsize=(20, 4), sharey=True)
fig.suptitle("Temperature 对概率分布的影响", fontsize=16, fontweight='bold')

for idx, T in enumerate(temperatures):
    probs = softmax_with_temperature(logits, T)
    axes[idx].bar(tokens, probs, color='steelblue', alpha=0.8)
    axes[idx].set_title(f"T = {T}", fontsize=14)
    axes[idx].set_ylim(0, 1)
    axes[idx].tick_params(axis='x', rotation=45)
    if idx == 0:
        axes[idx].set_ylabel("概率", fontsize=12)

plt.tight_layout()
plt.savefig("./images/temperature_effect.png", dpi=150, bbox_inches='tight')
plt.close()
print("✅ temperature_effect.png saved")
