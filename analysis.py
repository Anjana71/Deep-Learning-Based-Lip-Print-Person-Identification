import matplotlib.pyplot as plt

metrics = ['Accuracy','Precision','Recall','F1-score','Rank-1']
base = [99.6, 99.21, 99.43, 99.31, 0]
proposed = [0,0,0,0,93.04]

x = range(len(metrics))

plt.figure(figsize=(8,5))
plt.bar([i-0.2 for i in x], base, width=0.4, label='Base MobileNet')
plt.bar([i+0.2 for i in x], proposed, width=0.4, label='Proposed Siamese MobileViT')

plt.ylabel('Performance (%)')
plt.title('Evaluation Comparison')
plt.xticks(x, metrics)
plt.ylim(0,100)
plt.legend()
plt.grid(axis='y')

plt.tight_layout()
plt.savefig("evaluation_graph.png", dpi=300)
plt.show()