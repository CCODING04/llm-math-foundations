"""
Visualize: Entropy, KL Divergence, and Cross-Entropy
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# === Plot 1: Binary Entropy ===
p_vals = np.linspace(0.001, 0.999, 1000)
entropy = -p_vals * np.log2(p_vals) - (1 - p_vals) * np.log2(1 - p_vals)

axes[0].plot(p_vals, entropy, 'b-', linewidth=2)
axes[0].axhline(y=1, color='r', linestyle='--', alpha=0.5, label='Max Entropy = 1 bit')
axes[0].axvline(x=0.5, color='r', linestyle='--', alpha=0.5)
axes[0].set_xlabel('p(X=1)')
axes[0].set_ylabel('H(X) [bits]')
axes[0].set_title('Binary Entropy\n(Maximum at p=0.5)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# === Plot 2: KL Divergence Asymmetry ===
p_true = 0.7
q_vals = np.linspace(0.01, 0.99, 200)

kl_pq = p_true * np.log(p_true / q_vals) + (1-p_true) * np.log((1-p_true) / (1-q_vals))
kl_qp = q_vals * np.log(q_vals / p_true) + (1-q_vals) * np.log((1-q_vals) / (1-p_true))

axes[1].plot(q_vals, kl_pq, 'r-', linewidth=2, label=r'$D_{KL}(P\|Q)$, P=[0.7, 0.3]')
axes[1].plot(q_vals, kl_qp, 'b--', linewidth=2, label=r'$D_{KL}(Q\|P)$, P=[0.7, 0.3]')
axes[1].axvline(x=p_true, color='g', linestyle=':', alpha=0.7, label=f'P=Q (q={p_true})')
axes[1].set_xlabel('q (first component of Q)')
axes[1].set_ylabel('KL Divergence')
axes[1].set_title('KL Divergence Asymmetry\n' + r'$D_{KL}(P\|Q) \neq D_{KL}(Q\|P)$')
axes[1].legend()
axes[1].set_ylim(0, 3)
axes[1].grid(True, alpha=0.3)

# === Plot 3: Cross-Entropy = Entropy + KL Divergence ===
p_true_arr = np.array([0.7, 0.3])
q_vals2 = np.linspace(0.01, 0.99, 200)

H_P = -np.sum(p_true_arr * np.log2(p_true_arr))
cross_ent = np.array([
    -np.sum(p_true_arr * np.log2(np.array([q, 1-q])))
    for q in q_vals2
])
kl_div = cross_ent - H_P

axes[2].plot(q_vals2, cross_ent, 'r-', linewidth=2, label=r'$H(P,Q)$ Cross-Entropy')
axes[2].axhline(y=H_P, color='b', linestyle='--', linewidth=2, label=f'H(P) = {H_P:.3f}')
axes[2].fill_between(q_vals2, H_P, cross_ent, alpha=0.2, color='orange', label=r'$D_{KL}(P\|Q)$')
axes[2].axvline(x=0.7, color='g', linestyle=':', alpha=0.7, label='P=Q')
axes[2].set_xlabel('q (first component of Q)')
axes[2].set_ylabel('Information [bits]')
axes[2].set_title(r'Cross-Entropy = Entropy + KL Divergence' + '\n' + r'$H(P,Q) = H(P) + D_{KL}(P\|Q)$')
axes[2].legend()
axes[2].set_ylim(0, 3)
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
out = '/tmp/llm-math-foundations/chapter06-information-theory/images/entropy-binary.png'
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f"Saved: {out}")
