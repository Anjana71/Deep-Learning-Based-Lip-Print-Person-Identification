import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# -----------------------------
# LOAD MODEL
# -----------------------------
model = load_model("model/cheiloscopy_siamese_model.h5", compile=False)
encoder = model.get_layer("Encoder")

# -----------------------------
# LOAD DATA
# -----------------------------
X1 = np.load(r"C:\Users\ADMIN\Desktop\cheiloscopy\pairs\X1.npy")
X2 = np.load(r"C:\Users\ADMIN\Desktop\cheiloscopy\pairs\X2.npy")
y  = np.load(r"C:\Users\ADMIN\Desktop\cheiloscopy\pairs\y.npy")

# Fix channel issue if grayscale
if X1.shape[-1] == 1:
    X1 = np.repeat(X1, 3, axis=-1)
    X2 = np.repeat(X2, 3, axis=-1)

print("Computing embeddings...")

emb1 = encoder.predict(X1, batch_size=32)
emb2 = encoder.predict(X2, batch_size=32)

# -----------------------------
# DISTANCE FUNCTION
# -----------------------------
def euclidean(a, b):
    return np.linalg.norm(a - b)

# -----------------------------
# DISTANCE MATRIX (FOR RANKING)
# -----------------------------
print("Building distance matrix...")

dist_matrix = np.zeros((len(emb1), len(emb2)))

for i in range(len(emb1)):
    for j in range(len(emb2)):
        dist_matrix[i, j] = euclidean(emb1[i], emb2[j])

# -----------------------------
# RANK-K ACCURACY
# -----------------------------
def rank_k_accuracy(dist_matrix, y, k=1):
    correct = 0
    for i in range(len(dist_matrix)):
        ranked = np.argsort(dist_matrix[i])[:k]
        if np.any(y[ranked] == 1):
            correct += 1
    return correct / len(dist_matrix)

print("\n--- RANKING PERFORMANCE ---")
for k in [1, 3, 5, 10]:
    acc = rank_k_accuracy(dist_matrix, y, k)
    print(f"Rank-{k} Accuracy: {acc:.4f}")

# -----------------------------
# CMC CURVE
# -----------------------------
def compute_cmc(dist_matrix, y):
    cmc = np.zeros(len(dist_matrix[0]))
    
    for i in range(len(dist_matrix)):
        ranked = np.argsort(dist_matrix[i])
        for rank, idx in enumerate(ranked):
            if y[idx] == 1:
                cmc[rank:] += 1
                break

    cmc = cmc / len(dist_matrix)
    return cmc

cmc = compute_cmc(dist_matrix, y)

plt.figure()
plt.plot(cmc)
plt.title("CMC Curve")
plt.xlabel("Rank")
plt.ylabel("Accuracy")
plt.grid()
plt.show()

# -----------------------------
# DISTANCE ANALYSIS (FIXED)
# -----------------------------
same_dist = []
diff_dist = []

for i in range(len(emb1)):
    d = euclidean(emb1[i], emb2[i])
    if y[i] == 1:
        same_dist.append(d)
    else:
        diff_dist.append(d)

same_dist = np.array(same_dist)
diff_dist = np.array(diff_dist)

print("\n--- DISTANCE STATS ---")
print("Same Mean:", np.mean(same_dist))
print("Different Mean:", np.mean(diff_dist))

# Boxplot (BEST VISUAL)
plt.figure()
plt.boxplot([same_dist, diff_dist], labels=["Same", "Different"])
plt.title("Distance Comparison")
plt.ylabel("Distance")
plt.grid()
plt.show()

# -----------------------------
# THRESHOLD vs ACCURACY
# -----------------------------
print("\n--- THRESHOLD ANALYSIS ---")

scores = np.array([euclidean(emb1[i], emb2[i]) for i in range(len(emb1))])

thresholds = np.linspace(np.min(scores), np.max(scores), 50)
accuracies = []

for t in thresholds:
    preds = (scores < t).astype(int)
    acc = np.mean(preds == y)
    accuracies.append(acc)

best_acc = max(accuracies)
best_t = thresholds[np.argmax(accuracies)]

print(f"Best Accuracy: {best_acc:.4f}")
print(f"Best Threshold: {best_t:.4f}")

plt.figure()
plt.plot(thresholds, accuracies)
plt.title("Threshold vs Accuracy")
plt.xlabel("Threshold")
plt.ylabel("Accuracy")
plt.grid()
plt.show()

# -----------------------------
# TOP-K VISUALIZATION (STRONG PROOF)
# -----------------------------
def show_top_k(query_idx, k=5):
    distances = [
        euclidean(emb1[query_idx], emb2[j])
        for j in range(len(emb2))
    ]

    top_k = np.argsort(distances)[:k]

    plt.figure(figsize=(12,3))

    # Query
    plt.subplot(1, k+1, 1)
    plt.imshow(X1[query_idx].astype("uint8"))
    plt.title("Query")
    plt.axis("off")

    # Results
    for i, idx in enumerate(top_k):
        plt.subplot(1, k+1, i+2)
        plt.imshow(X2[idx].astype("uint8"))
        label = "Same" if y[idx] == 1 else "Diff"
        plt.title(f"Rank {i+1}\n{label}")
        plt.axis("off")

    plt.show()

# Show sample
show_top_k(0, k=5)

print("\n✅ FULL EVALUATION COMPLETED")