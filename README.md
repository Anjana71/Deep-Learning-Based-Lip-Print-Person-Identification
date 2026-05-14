# Cheiloscopy-Based Person Identification using Siamese MobileViT-S

## Overview

This project presents a deep learning–based forensic identification system using **Cheiloscopy**, the study of lip print patterns as biometric evidence. Similar to fingerprints, lip prints contain unique groove structures and texture patterns that can be used for reliable human identification.

The proposed framework utilizes a **Siamese Neural Network** integrated with a **MobileViT-S encoder** to learn discriminative feature embeddings from lip-related biometric samples. The system performs similarity learning between:

* Crime-scene lip print evidence
* Lip regions extracted from suspect facial images

By learning embedding similarity, the model ranks suspects according to their similarity score with the crime-scene lip print, enabling automated forensic person identification.

---

# Key Features

* Deep learning–based forensic biometric system
* Siamese similarity learning architecture
* MobileViT-S lightweight feature encoder
* Automated lip region extraction using MediaPipe
* Lip print enhancement using image preprocessing techniques
* Similarity-based suspect ranking
* Streamlit-based interactive interface
* Real-time suspect comparison and identification

---

# System Workflow

```text
Crime Scene Lip Print + Suspect Face Images
                    │
                    ▼
          Lip Region Extraction
         (MediaPipe Face Mesh)
                    │
                    ▼
         Lip Print Enhancement
     (CLAHE + Gaussian Filtering)
                    │
                    ▼
          Data Preprocessing
                    │
                    ▼
         Pair Generation Module
      (Positive / Negative Pairs)
                    │
                    ▼
        Siamese MobileViT-S Network
                    │
     ┌──────────────┴──────────────┐
     ▼                             ▼
 Feature Embedding 1      Feature Embedding 2
            │
            ▼
        L1 Distance
            │
            ▼
     Similarity Prediction
            │
            ▼
     Suspect Similarity Ranking
```

---

# Project Architecture

The proposed system consists of the following major modules:

## 1. Lip Region Extraction

* Extracts the lip region from suspect face images
* Implemented using **MediaPipe Face Mesh**
* Reduces irrelevant facial information

## 2. Lip Print Enhancement

* Enhances groove visibility and texture quality
* Uses:

  * CLAHE (Contrast Limited Adaptive Histogram Equalization)
  * Gaussian Blur
  * Normalization

## 3. Data Augmentation

To improve model generalization:

* Horizontal flipping
* Rotation
* Gaussian blur
* Normalization

## 4. Pair Generation

Creates:

* Positive pairs → same identity
* Negative pairs → different identities

Used for Siamese training.

## 5. Siamese MobileViT-S Network

The core deep learning framework:

* Shared MobileViT-S encoder
* 128-dimensional embeddings
* L1 distance similarity learning
* Sigmoid similarity classification

## 6. Similarity Ranking

Ranks suspects according to:

* embedding distance
* similarity score

Highest similarity indicates the most probable identity match.

---

# Dataset Structure

Each identity contains:

```text
dataset/
│
├── person_01/
│   ├── face.png
│   └── lip.png
│
├── person_02/
│   ├── face.png
│   └── lip.png
```

| File     | Description                        |
| -------- | ---------------------------------- |
| face.png | Face image used for lip extraction |
| lip.png  | Lip print image                    |

---

# Project Folder Structure

```text
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
├── requirements.txt
└── README.md
```

---

# Installation

## Create Virtual Environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / Mac

```bash
python3 -m venv venv
source venv/bin/activate
```

---

# Install Dependencies

```bash
pip install -r requirements.txt
```

If requirements file is unavailable:

```bash
pip install tensorflow opencv-python mediapipe numpy tqdm streamlit keras-cv-attention-models
```

---

# Execution Pipeline

---

## Step 1 — Preprocess Dataset

Extract lip regions and enhance lip print quality.

```bash
python preprocessing/run_preprocessing.py
```

### Output

```text
processed_data/
│
├── person_01/
│   ├── lip_print.npy
│   └── lip_region.npy
```

---

## Step 2 — Generate Training Pairs

Creates Siamese training pairs.

```bash
python pair_generation/run_pair_generation.py
```

### Output

```text
pairs/
│
├── X1.npy
├── X2.npy
└── y.npy
```

| File   | Description       |
| ------ | ----------------- |
| X1.npy | Lip print images  |
| X2.npy | Lip region images |
| y.npy  | Pair labels       |

---

## Step 3 — Train the Siamese Network

```bash
python model/train.py
```

### Output

```text
model/
│
├── cheiloscopy_siamese_model.h5
└── encoder_weights.h5
```

The encoder learns:

* 128-dimensional biometric embeddings
* similarity representations for identification

---

## Step 4 — Model Evaluation

```bash
python model/evaluate.py
```

### Example Result

```text
Rank-1 Accuracy: 89%
```

Meaning:

* Correct suspect appears at rank-1 in 89% of cases.

---

## Step 5 — Run Web Interface

```bash
streamlit run demo.py
```

The interface allows:

* Upload crime-scene lip print
* Upload suspect face images
* Perform automated suspect ranking
* Visualize similarity results

---

# Model Architecture

```text
Input Image (224×224×3)
            │
            ▼
     MobileViT-S Encoder
            │
            ▼
 Global Average Pooling
            │
            ▼
 Dense Embedding Layer
      (128-D Features)
            │
            ▼
      Feature Vector
```

Two images pass through:

* the same encoder network
* shared weights

Similarity is computed using:

* L1 distance
* Sigmoid similarity classifier

---

# Technologies Used

| Technology         | Purpose                 |
| ------------------ | ----------------------- |
| TensorFlow / Keras | Deep learning           |
| MobileViT-S        | Feature extraction      |
| OpenCV             | Image processing        |
| MediaPipe          | Lip landmark extraction |
| NumPy              | Numerical operations    |
| Streamlit          | Web interface           |

---

# Applications

* Forensic person identification
* Crime-scene investigation
* Biometric authentication
* Suspect ranking systems
* Automated forensic analysis

---

# Future Improvements

* Larger forensic lip print datasets
* Triplet loss optimization
* Hard-negative mining
* Attention-based similarity learning
* Multimodal biometric fusion
* Real-world latent evidence validation

---

# Results and Output Screenshots

The following screenshots demonstrate the functionality of the Streamlit-based forensic identification interface developed in this project.

---

## 1. Home Interface

Displays the main user interface for uploading:

* Crime-scene lip print
* Suspect face images

### Features

* Simple and interactive UI
* Multiple suspect image upload
* Real-time processing support

![Home Interface](Screenshots(216).png)

---

## 2. Crime-Scene Lip Print Upload

Shows the uploaded latent/crime-scene lip print image used as forensic evidence.

### Features

* Lip print preview
* Image preprocessing before inference
* Enhanced groove visibility

![Upload Interface](Screenshots(217).png)

---

## 3. Suspect Image Upload

Displays uploaded suspect face images used for identification.

### Features

* Multiple suspect support
* Automated lip region extraction
* Real-time image handling

![Upload Interface](Screenshots(218).png)

---

## 4. Similarity Ranking Output

Displays ranked suspects based on embedding similarity scores generated by the Siamese MobileViT-S model.

### Features

* Similarity score prediction
* Top matching suspect identification
* Ranking-based forensic comparison

### Example Output

```text
1. Suspect_03 → Similarity Score: 0.93
2. Suspect_07 → Similarity Score: 0.87
3. Suspect_01 → Similarity Score: 0.79
```



---

## 5. Final Identification Result

Shows the final predicted identity with the highest similarity score.

### Features

* Most probable suspect identification
* Automated forensic decision support
* Visual comparison output

![Final Result](Screenshots(219).png)

---

## 6. Complete Workflow Demonstration

Illustrates the end-to-end execution pipeline within the Streamlit application.

### Workflow

1. Upload crime-scene lip print
2. Upload suspect images
3. Process and extract lip regions
4. Generate embeddings
5. Compute similarity
6. Display ranked identification results

📷 *Add complete workflow screenshot here*

---

# Example Output Format

```text
Crime Scene Lip Print
        │
        ▼

Top Matching Suspects:
1. Person_12 → 0.93 Similarity
2. Person_07 → 0.88 Similarity
3. Person_21 → 0.81 Similarity
```

---

# Conclusion

This project demonstrates the feasibility of using deep learning and Siamese similarity learning for automated forensic person identification using lip prints. By integrating MobileViT-S embeddings, lip enhancement, and biometric similarity ranking, the system provides an efficient and scalable framework for AI-assisted forensic investigations.
