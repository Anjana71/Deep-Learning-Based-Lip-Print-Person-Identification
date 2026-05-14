import matplotlib.pyplot as plt
import numpy as np

features = [
    'Closed-set',
    'Open-set',
    'Cross-modal',
    'Ranked Output',
    'Real-time',
    'Forensic Use'
]

base = [1,0,0,0,1,0]
proposed = [1,1,1,1,1,1]

x = np.arange(len(features))

plt.figure(figsize=(9,5))
plt.bar(x-0.2, base, width=0.4, label='Base MobileNet')
plt.bar(x+0.2, proposed, width=0.4, label='Proposed Siamese MobileViT')

plt.xticks(x, features)
plt.yticks([0,1], ['No','Yes'])
plt.title('Capability Evaluation Comparison')
plt.legend()
plt.grid(axis='y')

plt.tight_layout()
plt.savefig("model_strength_comparison.png", dpi=300)
plt.show()