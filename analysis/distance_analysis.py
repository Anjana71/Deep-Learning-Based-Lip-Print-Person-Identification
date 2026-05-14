import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# -----------------------------
# Load model & encoder
# -----------------------------
model = load_model("model/cheiloscopy_siamese_model.h5", compile=False)
encoder = model.get_layer("Encoder")

# -----------------------------
# Load data
# -----------------------------
X1 = np.load(r"C:\Users\ADMIN\Desktop\cheiloscopy\pairs\X1.npy")
X2 = np.load(r"C:\Users\ADMIN\Desktop\cheiloscopy\pairs\X2.npy")
y  = np.load(r"C:\Users\ADMIN\Desktop\cheiloscopy\pairs\y.npy")

# Ensure RGB
if X1.shape[-1] == 1:
    X1 = np.repeat(X1, 3, axis=-1)
    X2 = np.repeat(X2, 3, axis=-1)

print("Computing embeddings...")

emb1 = encoder.predict(X1, batch_size=32)
emb2 = encoder.predict(X2, batch_size=32)

# -----------------------------
# Compute distances
# -----------------------------
same_dist = []
diff_dist = []

for i in range(len(emb1)):
    dist = np.linalg.norm(emb1[i] - emb2[i])  # pair distance
    
    if y[i] == 1:
        same_dist.append(dist)
    else:
        diff_dist.append(dist)

same_dist = np.array(same_dist)
diff_dist = np.array(diff_dist)

print("Same pairs:", len(same_dist))
print("Different pairs:", len(diff_dist))

# -----------------------------
# 1. NORMALIZED HISTOGRAM (FIXED)
# -----------------------------
plt.figure()
plt.hist(same_dist, bins=50, density=True, alpha=0.6, label="Same Person")
plt.hist(diff_dist, bins=50, density=True, alpha=0.6, label="Different Person")
plt.legend()
plt.title("Normalized Distance Distribution")
plt.xlabel("Distance")
plt.ylabel("Density")
plt.grid()
plt.show()

# -----------------------------
# 2. ZOOMED HISTOGRAM (VERY IMPORTANT)
# -----------------------------
plt.figure()
plt.hist(same_dist, bins=50, density=True, alpha=0.7, label="Same Person")
plt.hist(diff_dist, bins=50, density=True, alpha=0.7, label="Different Person")

# zoom into lower distance region
zoom_limit = np.percentile(diff_dist, 10)
plt.xlim(0, zoom_limit)

plt.legend()
plt.title("Zoomed Distance Distribution (Better Visualization)")
plt.xlabel("Distance")
plt.ylabel("Density")
plt.grid()
plt.show()

# -----------------------------
# 3. BOXPLOT (BEST FOR PPT)
# -----------------------------
plt.figure()
plt.boxplot([same_dist, diff_dist], labels=["Same Person", "Different Person"])
plt.ylabel("Distance")
plt.title("Distance Comparison (Boxplot)")
plt.grid()
plt.show()

print("✅ Distance analysis completed")