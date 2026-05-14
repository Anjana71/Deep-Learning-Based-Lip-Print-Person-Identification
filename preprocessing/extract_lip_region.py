import cv2
import numpy as np
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True
)

LIP_LANDMARKS = [
    61, 146, 91, 181, 84, 17,
    314, 405, 321, 375, 291, 308
]

def extract_lip_region(image_path):
    
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Unable to read image: {image_path}")

    h, w, _ = img.shape
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = mp_face_mesh.process(rgb)

    if not results.multi_face_landmarks:
        raise ValueError("No face detected")

    xs, ys = [], []
    for idx in LIP_LANDMARKS:
        lm = results.multi_face_landmarks[0].landmark[idx]
        xs.append(int(lm.x * w))
        ys.append(int(lm.y * h))

    x1, x2 = max(0, min(xs)), min(w, max(xs))
    y1, y2 = max(0, min(ys)), min(h, max(ys))

    lip_crop = img[y1:y2, x1:x2]
    lip_crop = cv2.resize(lip_crop, (224, 224))
    lip_crop = lip_crop.astype(np.float32) / 255.0

    return lip_crop
