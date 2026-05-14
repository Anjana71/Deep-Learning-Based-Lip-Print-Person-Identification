import numpy as np
import cv2
import random

def augment_image(img):
    # Resize to 224x224 if needed
    img = cv2.resize(img, (224, 224))

    # Ensure grayscale
    if img.ndim == 3 and img.shape[-1] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Horizontal flip
    if random.random() < 0.5:
        img = cv2.flip(img, 1)

    # Small rotation
    angle = random.uniform(-10, 10)
    h, w = img.shape
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1)
    img = cv2.warpAffine(img, M, (w, h))

    # Blur
    img = cv2.GaussianBlur(img, (3,3), 0)

    # Normalize to [0,1]
    img = img.astype(np.float32) / 255.0

    # Convert grayscale to RGB for MobileViT
    img = cv2.cvtColor((img*255).astype(np.uint8), cv2.COLOR_GRAY2RGB)  # shape (224,224,3)
    img = img.astype(np.float32) / 255.0

    return img
