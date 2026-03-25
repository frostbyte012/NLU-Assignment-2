import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# GRAPH 1: RNN PERFORMANCE COMPARISON
# ==========================================
models = ['Vanilla RNN', 'BiLSTM', 'Attention + RNN']
novelty = [88.00, 95.00, 93.50]
diversity = [94.50, 98.50, 96.50]

x = np.arange(len(models))
width = 0.35

fig, ax = plt.subplots(figsize=(8, 5))
rects1 = ax.bar(x - width/2, novelty, width, label='Novelty Rate (%)', color='#4C72B0')
rects2 = ax.bar(x + width/2, diversity, width, label='Diversity Rate (%)', color='#55A868')

ax.set_ylabel('Percentage (%)', fontsize=12)
ax.set_title('RNN Models Performance Comparison', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=11)
ax.set_ylim(80, 102) # Adjusted to highlight the differences at the top
ax.legend(loc='upper left')
ax.grid(axis='y', linestyle='--', alpha=0.7)

# Add text labels on top of the bars
for rects in [rects1, rects2]:
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}%',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig('RNN_Performance.png', dpi=300)
plt.close()
print("Saved: RNN_Performance.png")

# ==========================================
# GRAPH 2: PYTORCH "FROM SCRATCH" LOSSES
# ==========================================
epochs_pt = [1, 2, 3, 4, 5]
cbow_loss = [1.0793, 0.6921, 0.5074, 0.3757, 0.2780]
skipgram_loss = [8.0716, 8.5004, 8.3622, 8.3041, 8.2760]

plt.figure(figsize=(8, 5))
plt.plot(epochs_pt, cbow_loss, marker='o', linestyle='-', color='#4C72B0', linewidth=2.5, label='CBOW Loss')
plt.plot(epochs_pt, skipgram_loss, marker='s', linestyle='-', color='#C44E52', linewidth=2.5, label='Skip-gram Loss')

plt.title('PyTorch "From Scratch" Models Training Loss', fontsize=14, fontweight='bold')
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Cross-Entropy Loss', fontsize=12)
plt.xticks(epochs_pt)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.savefig('PyTorch_Loss.png', dpi=300)
plt.close()
print("Saved: PyTorch_Loss.png")

# ==========================================
# GRAPH 3: NUMPY "FROM SCRATCH" LOSS
# ==========================================
epochs_np = [10, 20, 30, 40, 50]
numpy_loss = [78705.32, 52340.23, 48762.07, 48262.56, 48124.06]

plt.figure(figsize=(8, 5))
plt.plot(epochs_np, numpy_loss, marker='^', linestyle='-', color='#8172B3', linewidth=2.5, label='NumPy Skip-gram Loss')

plt.title('NumPy Pure Math "From Scratch" Training Loss', fontsize=14, fontweight='bold')
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Cumulative Loss', fontsize=12)
plt.xticks(epochs_np)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.savefig('NumPy_Loss.png', dpi=300)
plt.close()
print("Saved: NumPy_Loss.png")
