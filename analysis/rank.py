import matplotlib.pyplot as plt

labels = ['Rank-1', 'Rank-5']
values = [0.9304, 0.9821]

plt.figure(figsize=(6,3))

for i, v in enumerate(values):
    plt.barh(labels[i], v)
    plt.text(v - 0.1, i, f"{v*100:.1f}%", va='center', color='white', fontweight='bold')

plt.xlim(0,1)
plt.title("Ranking Performance")
plt.xlabel("Accuracy")
plt.tight_layout()
plt.show()