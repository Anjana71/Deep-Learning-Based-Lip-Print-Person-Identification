import os
import numpy as np
from tqdm import tqdm

from preprocess_lip_print import preprocess_lip_print
from extract_lip_region import extract_lip_region

DATASET_DIR = r"C:\Users\ADMIN\Desktop\cheiloscopy\dataset"
OUTPUT_DIR = r"C:\Users\ADMIN\Desktop\cheiloscopy\processed_data"

os.makedirs(OUTPUT_DIR, exist_ok=True)

persons = sorted(os.listdir(DATASET_DIR))

for person in tqdm(persons, desc="Processing dataset"):
    person_path = os.path.join(DATASET_DIR, person)

    face_path = os.path.join(person_path, "face.png")
    lip_print_path = os.path.join(person_path, "lip.png")

    if not os.path.exists(face_path) or not os.path.exists(lip_print_path):
        print(f"Skipping {person}: Missing files")
        continue

    try:
        lip_print = preprocess_lip_print(lip_print_path)
        lip_region = extract_lip_region(face_path)

        out_person_dir = os.path.join(OUTPUT_DIR, person)
        os.makedirs(out_person_dir, exist_ok=True)

        np.save(os.path.join(out_person_dir, "lip_print.npy"), lip_print)
        np.save(os.path.join(out_person_dir, "lip_region.npy"), lip_region)

    except Exception as e:
        print(f"Error processing {person}: {e}")

print("Preprocessing completed successfully.")
