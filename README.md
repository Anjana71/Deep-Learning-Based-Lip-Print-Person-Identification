Cheiloscopy-Based Person Identification using Siamese MobileViT
Overview

This project implements a deep learning-based cheiloscopy recognition system for forensic person identification. Cheiloscopy refers to the study of lip print patterns, which are considered unique and stable biometric traits similar to fingerprints.

The proposed system uses a Siamese Neural Network with a MobileViT backbone to learn discriminative embeddings between:

Lip prints collected from crime scenes

Lip regions extracted from suspect face images

The model learns to measure similarity between pairs of images and ranks suspects based on similarity to the crime lip print.

Project Architecture

The system consists of the following stages:

Dataset
   │
   ▼
Preprocessing
   │
   ├── Lip Region Extraction (MediaPipe Face Mesh)
   └── Lip Print Enhancement (CLAHE + Gaussian Blur)
   │
   ▼
Processed Dataset (.npy files)
   │
   ▼
Data Augmentation
   │
   ▼
Pair Generation (Positive / Negative pairs)
   │
   ▼
Siamese Neural Network
   │
   ├── MobileViT Encoder
   ├── Feature Embedding (128D)
   └── L1 Distance Similarity
   │
   ▼
Training
   │
   ▼
Model Evaluation
   │
   ▼
Ranking Based Suspect Identification
Dataset Structure

Each identity must contain two images:

dataset/
│
├── person_01/
│   ├── face.png
│   └── lip.png
│
├── person_02/
│   ├── face.png
│   └── lip.png
│
├── person_03/
│   ├── face.png
│   └── lip.png

Where:

File	Description
face.png	Face photograph used to extract lip region
lip.png	Lip print image
Project Folder Structure
cheiloscopy/
│
├── dataset/
│
├── preprocessing/
│   ├── extract_lip_region.py
│   ├── preprocess_lip_print.py
│   └── run_preprocessing.py
│
├── pair_generation/
│   ├── augment.py
│   ├── generate_pairs.py
│   └── run_pair_generation.py
│
├── processed_data/
│
├── pairs/
│   ├── X1.npy
│   ├── X2.npy
│   └── y.npy
│
├── model/
│   ├── siamese_model.py
│   ├── train.py
│   ├── evaluate.py
│   ├── rank_evaluate.py
│   ├── encoder_weights.h5
│   └── cheiloscopy_siamese_model.h5
│
├── demo.py
│
├── requirements.txt
└── README.md
Installation

Create a virtual environment.

Windows
python -m venv venv
venv\Scripts\activate
Linux / Mac
python3 -m venv venv
source venv/bin/activate
Install Dependencies
pip install -r requirements.txt

If requirements file is not present:

pip install tensorflow opencv-python mediapipe numpy tqdm streamlit keras-cv-attention-models
Execution Pipeline

The project must be executed in four stages.

Step 1: Preprocess Dataset

Extract lip regions and enhance lip prints.

Run:

python preprocessing/run_preprocessing.py

Output:

processed_data/
   person_01/
       lip_print.npy
       lip_region.npy
Step 2: Generate Training Pairs

Creates positive and negative image pairs with augmentation.

Run:

python pair_generation/run_pair_generation.py

Output:

pairs/
   X1.npy
   X2.npy
   y.npy

Where:

File	Description
X1.npy	Lip print images
X2.npy	Lip region images
y.npy	Labels (1 = same person, 0 = different person)
Step 3: Train the Siamese Network
Run:

python model/train.py
Output:
model/
   cheiloscopy_siamese_model.h5
   encoder_weights.h5

The encoder learns 128-dimensional feature embeddings for lip images.

Step 4: Evaluate the Model

Run classification evaluation:

Rank-1 Accuracy: 0.89
This means the correct suspect appears first in the ranking 89% of the time.

Step 6: Run the Web Interface
A simple GUI is implemented using Streamlit.

Run:

streamlit run demo.py
The interface allows the user to:
Upload crime lip print
Upload suspect images
Perform similarity ranking
Output shows ranked suspect matches based on similarity distance.

Model Architecture
The system uses a Siamese Neural Network consisting of:

Input Image (224x224x3)
        │
        ▼
MobileViT Encoder
        │
        ▼
Global Average Pooling
        │
        ▼
Dense Layer (128 Embedding)
        │
        ▼
Feature Vector

Two images pass through the same encoder.
Similarity is computed using:
L1 Distance
Followed by:
Sigmoid classifier
Data Augmentation

To overcome small dataset size, augmentation is applied:
Horizontal flipping
Rotation
Gaussian blur
Normalization

Applications

This system can be used in:
Forensic investigation
Crime scene analysis
Biometric authentication
Identity verification
Future Improvements

Possible improvements include:

Larger cheiloscopy datasets
Triplet loss training
Hard-negative mining
Advanced augmentation
Multimodal biometric fusion