import numpy as np
from tensorflow.keras.models import load_model

model = load_model("model/cheiloscopy_siamese_model.h5")

X1 = np.load(r"C:\Users\ADMIN\Desktop\cheiloscopy\pairs\X1.npy")
X2 = np.load(r"C:\Users\ADMIN\Desktop\cheiloscopy\pairs\X2.npy")
y  = np.load(r"C:\Users\ADMIN\Desktop\cheiloscopy\pairs\y.npy")

# Ensure RGB
if X1.shape[-1] == 1:
    X1 = np.repeat(X1, 3, axis=-1)
    X2 = np.repeat(X2, 3, axis=-1)

loss, acc = model.evaluate([X1, X2], y, batch_size=16)
print("Evaluation Accuracy:", acc)
