"""
RMSNorm 与权重初始化演示
- RMSNorm 效果可视化
- Xavier / He 初始化的正交性分析
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ==========================================
# Part 1: RMSNorm 演示
# ==========================================
print("=" * 50)
print("Part 1: RMSNorm 归一化")
print("=" * 50)

np.random.seed(42)
d = 512  # 隐藏维度

# 模拟不同尺度的输入向量
x_small = np.random.randn(d) * 0.1    # 小尺度
x_medium = np.random.randn(d) * 1.0   # 中等尺度
x_large = np.random.randn(d) * 10.0   # 大尺度

def rms_norm(x, gamma=None, epsilon=1e-8):
    """RMSNorm: x / RMS(x) * gamma"""
    rms = np.sqrt(np.mean(x ** 2) + epsilon)
    x_normed = x / rms
    if gamma is not None:
        x_normed = x_normed * gamma
    return x_normed, rms

gamma = np.ones(d)  # 可学习参数，初始化为1

print(f"输入统计:")
for name, x in [("小尺度", x_small), ("中等", x_medium), ("大尺度", x_large)]:
    rms_before = np.sqrt(np.mean(x**2))
    x_normed, rms_after = rms_norm(x, gamma)
    rms_after_norm = np.sqrt(np.mean(x_normed**2))
    print(f"  {name}: RMS={rms_before:.4f} → 归一化后 RMS={rms_after_norm:.4f}")

# 可视化
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 归一化前的分布
ax1 = axes[0, 0]
for name, x, color in [("小尺度", x_small, 'blue'), ("中等", x_medium, 'green'), ("大尺度", x_large, 'red')]:
    ax1.hist(x, bins=50, alpha=0.5, color=color, label=f'{name} (σ={np.std(x):.2f})')
ax1.set_title('RMSNorm 之前：不同尺度的输入', fontsize=13)
ax1.legend(fontsize=10)
ax1.set_xlabel('值')

# 归一化后的分布
ax2 = axes[0, 1]
for name, x, color in [("小尺度", x_small, 'blue'), ("中等", x_medium, 'green'), ("大尺度", x_large, 'red')]:
    x_normed, _ = rms_norm(x, gamma)
    ax2.hist(x_normed, bins=50, alpha=0.5, color=color, label=f'{name} (归一化后)')
ax2.set_title('RMSNorm 之后：所有输入归一化到相同尺度', fontsize=13)
ax2.legend(fontsize=10)
ax2.set_xlabel('值')

# Part 2: 权重初始化对比
print("\n" + "=" * 50)
print("Part 2: 权重初始化")
print("=" * 50)

n_in, n_out = 512, 512

# Xavier 初始化
W_xavier = np.random.randn(n_in, n_out) * np.sqrt(2.0 / (n_in + n_out))
# He 初始化
W_he = np.random.randn(n_in, n_out) * np.sqrt(2.0 / n_in)
# 随机初始化（无缩放）
W_naive = np.random.randn(n_in, n_out) * 0.01
# 正交初始化
Q, _ = np.linalg.qr(np.random.randn(n_in, n_out))

for name, W in [("Xavier", W_xavier), ("He", W_he), ("Naive(0.01)", W_naive), ("正交", Q)]:
    singular_vals = np.linalg.svd(W, compute_uv=False)
    print(f"  {name:10s}: σ_max={singular_vals[0]:.4f}, σ_min={singular_vals[-1]:.4f}, "
          f"比值={singular_vals[0]/singular_vals[-1]:.2f}")

# 可视化初始化方法的奇异值分布
ax3 = axes[1, 0]
for name, W, color in [("Xavier", W_xavier, 'blue'), ("He", W_he, 'green'),
                         ("Naive(0.01)", W_naive, 'red'), ("正交", Q, 'purple')]:
    sv = np.linalg.svd(W, compute_uv=False)
    ax3.plot(range(len(sv)), sv, alpha=0.7, color=color, label=name)
ax3.set_title('不同初始化方法的奇异值分布', fontsize=13)
ax3.set_xlabel('奇异值序号')
ax3.set_ylabel('奇异值大小')
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3)

# 正交初始化 vs 随机初始化 的信号传播
ax4 = axes[1, 1]
depths = range(1, 51)
for name, W, color in [("Xavier", W_xavier, 'blue'), ("正交", Q, 'purple'),
                         ("Naive(0.01)", W_naive, 'red')]:
    norms = []
    x = np.random.randn(n_in)
    x = x / np.linalg.norm(x)
    for d in depths:
        x = W @ x
        # 非线性（简化 ReLU）
        x = np.maximum(x, 0)
        x_norm = np.linalg.norm(x)
        norms.append(x_norm)
        x = x / (x_norm + 1e-8)  # 防止数值溢出
    ax4.plot(depths, norms, alpha=0.7, color=color, label=name)

ax4.set_title('不同初始化下的信号传播（50层）', fontsize=13)
ax4.set_xlabel('层深度')
ax4.set_ylabel('输出范数')
ax4.legend(fontsize=10)
ax4.grid(True, alpha=0.3)
ax4.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('images/rmsnorm_init.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n✓ 已保存: images/rmsnorm_init.png")

print("\n✅ 所有演示完成！")
