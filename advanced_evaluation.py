import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve, auc

# -----------------------------
# Distance Matrix
# -----------------------------
def compute_distance_matrix(query_emb, gallery_emb):
    dist_matrix = np.zeros((len(query_emb), len(gallery_emb)))
    
    for i in range(len(query_emb)):
        for j in range(len(gallery_emb)):
            dist_matrix[i, j] = np.linalg.norm(query_emb[i] - gallery_emb[j])
    
    return dist_matrix


# -----------------------------
# Rank-K Accuracy
# -----------------------------
def rank_k_accuracy(dist_matrix, k=1):
    correct = 0
    
    for i in range(len(dist_matrix)):
        sorted_idx = np.argsort(dist_matrix[i])
        top_k = sorted_idx[:k]
        
        if i in top_k:  # correct match is same index
            correct += 1
    
    return correct / len(dist_matrix)

# -----------------------------
# CMC Curve
# -----------------------------
def compute_cmc(dist_matrix):
    ranks = np.zeros(len(dist_matrix))
    
    for i in range(len(dist_matrix)):
        sorted_idx = np.argsort(dist_matrix[i])
        rank = np.where(sorted_idx == i)[0][0]
        
        ranks[rank:] += 1
    
    return ranks / len(dist_matrix)


# -----------------------------
# Precision, Recall, F1
# -----------------------------
def compute_classification_metrics(dist_matrix, threshold=0.5):
    y_true = []
    y_pred = []
    
    for i in range(len(dist_matrix)):
        for j in range(len(dist_matrix)):
            y_true.append(1 if i == j else 0)
            y_pred.append(1 if dist_matrix[i][j] < threshold else 0)
    
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    return precision, recall, f1


# -----------------------------
# ROC + AUC + EER
# -----------------------------
def compute_roc_eer(dist_matrix):
    y_true = []
    scores = []
    
    for i in range(len(dist_matrix)):
        for j in range(len(dist_matrix)):
            y_true.append(1 if i == j else 0)
            scores.append(-dist_matrix[i][j])  # invert distance
    
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    roc_auc = auc(fpr, tpr)

    # EER
    fnr = 1 - tpr
    eer = fpr[np.nanargmin(np.abs(fnr - fpr))]
    
    return fpr, tpr, roc_auc, eer


# -----------------------------
# mAP
# -----------------------------
def compute_map(dist_matrix):
    APs = []
    
    for i in range(len(dist_matrix)):
        sorted_idx = np.argsort(dist_matrix[i])
        correct = 0
        precision_sum = 0
        
        for rank, idx in enumerate(sorted_idx):
            if idx == i:
                correct += 1
                precision_sum += correct / (rank + 1)
        
        APs.append(precision_sum / correct if correct > 0 else 0)
    
    return np.mean(APs)


# -----------------------------
# Distance Distribution Plot
# -----------------------------
def plot_distance_distribution(dist_matrix):
    same = []
    diff = []
    
    for i in range(len(dist_matrix)):
        for j in range(len(dist_matrix)):
            if i == j:
                same.append(dist_matrix[i][j])
            else:
                diff.append(dist_matrix[i][j])
    
    plt.hist(same, bins=50, alpha=0.5, label="Same Person")
    plt.hist(diff, bins=50, alpha=0.5, label="Different Person")
    plt.legend()
    plt.title("Distance Distribution")
    plt.show()