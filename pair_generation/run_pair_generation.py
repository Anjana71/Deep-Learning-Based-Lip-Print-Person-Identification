import numpy as np
from generate_pairs import load_data, generate_pairs

PROCESSED_DIR = r"C:\Users\ADMIN\Desktop\cheiloscopy\processed_data"
OUTPUT_DIR = r"C:\Users\ADMIN\Desktop\cheiloscopy\pairs"

import os
os.makedirs(OUTPUT_DIR, exist_ok=True)

data = load_data(PROCESSED_DIR)
X1, X2, y = generate_pairs(data, augmentations=8)

np.save(f"{OUTPUT_DIR}/X1.npy", X1)
np.save(f"{OUTPUT_DIR}/X2.npy", X2)
np.save(f"{OUTPUT_DIR}/y.npy", y)

print("Step 3 completed")
print("Pairs created:", len(y))
