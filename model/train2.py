# train_siamese.py
import numpy as np
import os
from sklearn.model_selection import train_test_split
from siamese_model import build_siamese_network

# -----------------------------
# Load pairs of images
# -----------------------------
X1 = np.load(r"C:\Users\ADMIN\Desktop\cheiloscopy\pairs\X1.npy")
X2 = np.load(r"C:\Users\ADMIN\Desktop\cheiloscopy\pairs\X2.npy")
y  = np.load(r"C:\Users\ADMIN\Desktop\cheiloscopy\pairs\y.npy")

print("Total samples:", len(y))
print("Label distribution:", np.unique(y, return_counts=True))

# -----------------------------
# Convert grayscale (1-channel) to RGB (3-channel)
# -----------------------------
if X1.shape[-1] == 1:
    X1 = np.repeat(X1, 3, axis=-1)
    X2 = np.repeat(X2, 3, axis=-1)

print("X1 shape (RGB):", X1.shape)
print("X2 shape (RGB):", X2.shape)

# -----------------------------
# 🔥 IMPORTANT: Shuffle BEFORE split
# -----------------------------
indices = np.arange(len(y))
np.random.shuffle(indices)

X1 = X1[indices]
X2 = X2[indices]
y  = y[indices]

# -----------------------------
# ✅ Stratified train/validation split (BEST PRACTICE)
# -----------------------------
X1_train, X1_val, X2_train, X2_val, y_train, y_val = train_test_split(
    X1,
    X2,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

print("Train size:", len(y_train))
print("Val size:", len(y_val))
print("Train distribution:", np.unique(y_train, return_counts=True))
print("Val distribution:", np.unique(y_val, return_counts=True))

# -----------------------------
# Build Siamese model
# -----------------------------
model = build_siamese_network(input_shape=(224, 224, 3))
model.summary()

# -----------------------------
# Training
# -----------------------------
history = model.fit(
    [X1_train, X2_train],
    y_train,
    batch_size=16,
    epochs=20,
    validation_data=([X1_val, X2_val], y_val),
    shuffle=True,
    verbose=1
)

# -----------------------------
# Save model and encoder weights
# -----------------------------
os.makedirs("model", exist_ok=True)

model.save("model/cheiloscopy_siamese_model_new.h5")

encoder = model.get_layer("Encoder")
encoder.save_weights("model/encoder_weights.h5")

print("✅ Training completed and encoder weights saved")