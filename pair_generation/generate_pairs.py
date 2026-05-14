import os
import numpy as np
import random
from augment import augment_image

def load_data(processed_dir):
    data = {}
    for person in os.listdir(processed_dir):
        path = os.path.join(processed_dir, person)
        lp = np.load(os.path.join(path, "lip_print.npy"))
        lr = np.load(os.path.join(path, "lip_region.npy"))
        data[person] = (lp, lr)
    return data

def generate_pairs(data, augmentations=5):
    X1, X2, y = [], [], []
    persons = list(data.keys())

    for person in persons:
        lp, lr = data[person]

        # Positive pairs
        for _ in range(augmentations):
            X1.append(augment_image(lp))
            X2.append(augment_image(lr))
            y.append(1)

        # Negative pairs
        for _ in range(augmentations):
            neg_person = random.choice([p for p in persons if p != person])
            _, neg_lr = data[neg_person]
            X1.append(augment_image(lp))
            X2.append(augment_image(neg_lr))
            y.append(0)

    return np.array(X1), np.array(X2), np.array(y)
