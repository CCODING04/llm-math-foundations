"""
Visualize: Cross-Entropy Loss and Perplexity During LLM Training
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

steps = np.arange(1, 101)
np.random.seed(42)

# Simulated train/val losses
base_loss = 5.0 * np.exp(-0.03 * steps) + 1.5
train_loss = base_loss + np.random.normal(0, 0.1, len(steps))
val_loss = 5.0 * np.exp(-0.025 * steps) + 1.8 + np.random.normal(0, 0.08, len(steps))

train_ppl = np.exp(train_loss)
val_ppl = np.exp(val_loss)

# Plot 1: Cross-Entropy Loss
axes[0].plot(steps, train_loss, 'b-', alpha=0.7, label='Train Loss')
axes[0].plot(steps, val_loss, 'r-', alpha=0.7, label='Val Loss')
axes[0].set_xlabel('Training Step')
axes[0].set_ylabel('Cross-Entropy Loss')
axes[0].set_title('Cross-Entropy Loss During Training')
axes[0].legend()
axes[0].grid(True, alpha=0.3)
axes[0].annotate(f'Start: {train_loss[0]:.2f}\nPPL: {train_ppl[0]:.1f}',
                 xy=(5, train_loss[5]), fontsize=9,
                 bbox=dict(boxstyle='round', facecolor='lightyellow'))
axes[0].annotate(f'End: {train_loss[-1]:.2f}\nPPL: {train_ppl[-1]:.1f}',
                 xy=(75, train_loss[-1]+0.3), fontsize=9,
                 bbox=dict(boxstyle='round', facecolor='lightgreen'))

# Plot 2: Perplexity
axes[1].plot(steps, train_ppl, 'b-', alpha=0.7, label='Train PPL')
axes[1].plot(steps, val_ppl, 'r-', alpha=0.7, label='Val PPL')
axes[1].set_xlabel('Training Step')
axes[1].set_ylabel('Perplexity (PPL)')
axes[1].set_title('Perplexity = exp(Cross-Entropy Loss)')
axes[1].set_yscale('log')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

axes[1].axhspan(1, 10, alpha=0.1, color='green')
axes[1].axhspan(10, 50, alpha=0.1, color='yellow')
axes[1].axhspan(50, 500, alpha=0.1, color='red')
axes[1].text(95, 5, 'Excellent', fontsize=8, ha='right', color='green')
axes[1].text(95, 30, 'Fair', fontsize=8, ha='right', color='orange')
axes[1].text(95, 150, 'Poor', fontsize=8, ha='right', color='red')

plt.tight_layout()
out = '/tmp/llm-math-foundations/chapter06-information-theory/images/cross-entropy-kl.png'
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f"Saved: {out}")
