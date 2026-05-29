"""
信息论基础演示：计算交叉熵损失（模拟 LLM 训练）
"""
import numpy as np

def softmax(x):
    e_x = np.exp(x - np.max(x))  # 数值稳定
    return e_x / e_x.sum()

# === 模拟一个简单的 LLM 预测 ===
vocab = ["我", "爱", "吃", "苹", "果"]

# 模型预测的 logits
logits = np.array([2.0, 1.0, 0.5, 0.3, 0.2])
q = softmax(logits)
print("模型预测分布 Q:", dict(zip(vocab, q.round(4))))

# 真实 token 是 "我"（index=0）
true_token_idx = 0
p = np.zeros(5)
p[true_token_idx] = 1.0  # one-hot

# 手动计算交叉熵损失
cross_entropy = -np.sum(p * np.log(q))
print(f"\n交叉熵损失: {cross_entropy:.4f}")
print(f"等价于: -log(q[真token]) = -log({q[true_token_idx]:.4f}) = {-np.log(q[true_token_idx]):.4f}")

# 计算困惑度
ppl = np.exp(cross_entropy)
print(f"困惑度 (PPL): {ppl:.4f}")

# === 不同预测质量对比 ===
print("\n--- 不同预测质量的对比 ---")
scenarios = {
    "完美预测 (logits=[100,0,0,0,0])": np.array([100, 0, 0, 0, 0]),
    "较好预测": np.array([2.0, 1.0, 0.5, 0.3, 0.2]),
    "随机预测": np.array([0.1, 0.1, 0.1, 0.1, 0.1]),
    "反向预测 (真token概率最低)": np.array([0.1, 0.2, 0.3, 0.5, 2.0]),
}

for name, logits in scenarios.items():
    q = softmax(logits)
    loss = -np.log(q[true_token_idx])
    ppl = np.exp(loss)
    print(f"  {name}: 损失={loss:.4f}, PPL={ppl:.4f}")
