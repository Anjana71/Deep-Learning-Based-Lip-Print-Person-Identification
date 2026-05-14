# train_siamese.py
import numpy as np
import os
from siamese_model import build_siamese_network

# -----------------------------
# Load pairs of images
# -----------------------------
X1 = np.load(r"C:\Users\ADMIN\Desktop\cheiloscopy\pairs\X1.npy")
X2 = np.load(r"C:\Users\ADMIN\Desktop\cheiloscopy\pairs\X2.npy")
y  = np.load(r"C:\Users\ADMIN\Desktop\cheiloscopy\pairs\y.npy")

# -----------------------------
# Convert grayscale (1-channel) to RGB (3-channel) for MobileViT
# -----------------------------
if X1.shape[-1] == 1:
    X1 = np.repeat(X1, 3, axis=-1)
    X2 = np.repeat(X2, 3, axis=-1)

print("X1 shape (RGB):", X1.shape)
print("X2 shape (RGB):", X2.shape)

# -----------------------------
# Build Siamese model
# -----------------------------
model = build_siamese_network(input_shape=(224, 224, 3))
model.summary()

# -----------------------------
# Training
# -----------------------------
history = model.fit(
    [X1, X2],
    y,
    batch_size=16,
    epochs=20,
    validation_split=0.2,
    shuffle=True
)

# -----------------------------
# Save model and encoder weights
# -----------------------------
os.makedirs("model", exist_ok=True)

# Save full Siamese model
model.save("model/cheiloscopy_siamese_model.h5")

# Save encoder separately
encoder = model.get_layer("Encoder")
encoder.save_weights("model/encoder_weights_n.h5")

print("✅ Training completed and encoder weights saved")
