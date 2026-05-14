import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve, auc

# -----------------------------
# Load Model
# -----------------------------
model = load_model("model/cheiloscopy_siamese_model.h5", compile=False)
encoder = model.get_layer("Encoder")

# -----------------------------
# Load Data
# -----------------------------
X1 = np.load(r"C:\Users\ADMIN\Desktop\cheiloscopy\pairs\X1.npy")
X2 = np.load(r"C:\Users\ADMIN\Desktop\cheiloscopy\pairs\X2.npy")
y  = np.load(r"C:\Users\ADMIN\Desktop\cheiloscopy\pairs\y.npy")

# -----------------------------
# Ensure RGB (MobileViT requirement)
# -----------------------------
if X1.shape[-1] == 1:
    X1 = np.repeat(X1, 3, axis=-1)
    X2 = np.repeat(X2, 3, axis=-1)

print("Computing embeddings...")

# -----------------------------
# Embeddings
# -----------------------------
embeddings_X1 = encoder.predict(X1, batch_size=32)
embeddings_X2 = encoder.predict(X2, batch_size=32)

# -----------------------------
# Distance Matrix
# -----------------------------
def compute_distance_matrix(q, g):
    dist = np.zeros((len(q), len(g)))
    for i in range(len(q)):
        for j in range(len(g)):
            dist[i, j] = np.linalg.norm(q[i] - g[j])
    return dist

dist_matrix = compute_distance_matrix(embeddings_X1, embeddings_X2)

print("Evaluation started...")

# -----------------------------
# Rank-1 & Rank-5 (Correct for pair-based data)
# -----------------------------
def rank_k_accuracy(dist_matrix, y, k=1):
    correct = 0
    total = 0
    
    for i in range(len(dist_matrix)):
        if y[i] == 1:   # evaluate only true matches
            total += 1
            sorted_idx = np.argsort(dist_matrix[i])
            top_k = sorted_idx[:k]
            
            if any(y[idx] == 1 for idx in top_k):
                correct += 1
    
    return correct / total if total > 0 else 0

rank1 = rank_k_accuracy(dist_matrix, y, k=1)
rank5 = rank_k_accuracy(dist_matrix, y, k=5)

print(f"Rank-1 Accuracy: {rank1:.4f}")
print(f"Rank-5 Accuracy: {rank5:.4f}")

# -----------------------------
# CMC Curve
# -----------------------------
def compute_cmc(dist_matrix, y):
    ranks = np.zeros(len(dist_matrix))
    total = 0
    
    for i in range(len(dist_matrix)):
        if y[i] != 1:
            continue
        
        total += 1
        sorted_idx = np.argsort(dist_matrix[i])
        
        for rank, idx in enumerate(sorted_idx):
            if y[idx] == 1:
                ranks[rank:] += 1
                break
    
    return ranks / total if total > 0 else ranks

cmc = compute_cmc(dist_matrix, y)

plt.plot(cmc)
plt.title("CMC Curve")
plt.xlabel("Rank")
plt.ylabel("Accuracy")
plt.grid()
plt.show()

# -----------------------------
# Classification Metrics
# -----------------------------
scores = model.predict([X1, X2])
y_pred = (scores > 0.5).astype(int)

precision = precision_score(y, y_pred)
recall = recall_score(y, y_pred)
f1 = f1_score(y, y_pred)

print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)

# -----------------------------
# ROC + AUC + EER
# -----------------------------
fpr, tpr, thresholds = roc_curve(y, scores)
roc_auc = auc(fpr, tpr)

fnr = 1 - tpr
eer = fpr[np.nanargmin(np.abs(fnr - fpr))]

print("AUC:", roc_auc)
print("EER:", eer)

plt.plot(fpr, tpr)
plt.title("ROC Curve")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.grid()
plt.show()

# -----------------------------
# mAP (Mean Average Precision)
# -----------------------------
def compute_map(dist_matrix, y):
    APs = []
    
    for i in range(len(dist_matrix)):
        if y[i] != 1:
            continue
        
        distances = dist_matrix[i]
        sorted_idx = np.argsort(distances)
        
        correct = 0
        precision_sum = 0
        
        for rank, idx in enumerate(sorted_idx):
            if y[idx] == 1:
                correct += 1
                precision_sum += correct / (rank + 1)
        
        if correct > 0:
            APs.append(precision_sum / correct)
    
    return np.mean(APs) if len(APs) > 0 else 0

map_score = compute_map(dist_matrix, y)
print("mAP:", map_score)

# -----------------------------
# Distance Distribution
# -----------------------------
same = []
diff = []

for i in range(len(dist_matrix)):
    for j in range(len(dist_matrix)):
        if y[i] == 1 and y[j] == 1:
            same.append(dist_matrix[i][j])
        else:
            diff.append(dist_matrix[i][j])

plt.hist(same, bins=50, alpha=0.5, label="Same Person")
plt.hist(diff, bins=50, alpha=0.5, label="Different Person")
plt.legend()
plt.title("Distance Distribution")
plt.show()

print("✅ Evaluation Completed Successfully")