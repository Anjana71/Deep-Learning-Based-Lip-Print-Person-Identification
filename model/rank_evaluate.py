import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load trained siamese model
model = load_model("model/cheiloscopy_siamese_model.h5", compile=False)

# Extract encoder
encoder = model.get_layer("Encoder")

# Load data
X1 = np.load(r"C:\Users\ADMIN\Desktop\cheiloscopy\pairs\X1.npy")
X2 = np.load(r"C:\Users\ADMIN\Desktop\cheiloscopy\pairs\X2.npy")
y  = np.load(r"C:\Users\ADMIN\Desktop\cheiloscopy\pairs\y.npy")

# Add channel dimension
X1 = X1[..., np.newaxis]
X2 = X2[..., np.newaxis]

print("Computing embeddings...")

# ✅ PREDICT ONCE
embeddings_X1 = encoder.predict(X1, batch_size=32)
embeddings_X2 = encoder.predict(X2, batch_size=32)

print("Ranking evaluation started...")

def euclidean(a, b):
    return np.linalg.norm(a - b)

correct = 0

for i in range(len(embeddings_X1)):
    distances = [
        euclidean(embeddings_X1[i], embeddings_X2[j])
        for j in range(len(embeddings_X2))
    ]
    predicted_index = np.argmin(distances)
    if y[predicted_index] == 1:
        correct += 1

rank_accuracy = correct / len(embeddings_X1)
print(f"Rank-1 Accuracy: {rank_accuracy:.4f}")
