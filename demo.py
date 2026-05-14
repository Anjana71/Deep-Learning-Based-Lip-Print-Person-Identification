import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model

# -------------------------------------------------
# App Title
# -------------------------------------------------
st.set_page_config(page_title="Cheiloscopy Identification", layout="centered")
st.title("🔍 Cheiloscopy Crime Lip Identification System")

# -------------------------------------------------
# Load Model
# -------------------------------------------------
@st.cache_resource
def load_siamese_model():
    model = load_model(
        r"C:\Users\ADMIN\Desktop\cheiloscopy\model\cheiloscopy_siamese_model.h5",
        compile=False
    )
    return model, model.get_layer("Encoder")

with st.spinner("Loading trained model..."):
    model, encoder = load_siamese_model()

st.success("✅ Model loaded successfully")

# -------------------------------------------------
# Upload Crime Lip Print
# -------------------------------------------------
st.subheader("1️⃣ Upload Crime Scene Lip Print")

crime_file = st.file_uploader(
    "Upload crime lip print image",
    type=["jpg", "png", "jpeg"]
)

crime_img = None
if crime_file:
    crime_img = cv2.imdecode(
        np.frombuffer(crime_file.read(), np.uint8),
        cv2.IMREAD_GRAYSCALE
    )
    crime_img = cv2.resize(crime_img, (224, 224))
    crime_img = crime_img.astype(np.float32) / 255.0

    # 🔥 Convert to RGB
    crime_img = cv2.cvtColor(
        (crime_img * 255).astype(np.uint8),
        cv2.COLOR_GRAY2RGB
    )
    crime_img = crime_img.astype(np.float32) / 255.0

    crime_img = crime_img[np.newaxis, ...]  # (1,224,224,3)

    st.image(crime_img[0], caption="Crime Lip Print", width=200)

# -------------------------------------------------
# Number of Suspects
# -------------------------------------------------
st.subheader("2️⃣ Suspect Information")

num_suspects = st.number_input(
    "Enter number of suspects",
    min_value=1,
    max_value=20,
    step=1
)

# -------------------------------------------------
# Upload Suspect Images
# -------------------------------------------------
suspect_images = None

if crime_img is not None and num_suspects > 0:
    suspect_files = st.file_uploader(
        f"Upload exactly {num_suspects} suspect images",
        type=["jpg", "png", "jpeg"],
        accept_multiple_files=True
    )

    if suspect_files:
        if len(suspect_files) != num_suspects:
            st.warning(f"⚠ Please upload exactly {num_suspects} images")
        else:
            suspect_images = []
            for f in suspect_files:
                img = cv2.imdecode(
                    np.frombuffer(f.read(), np.uint8),
                    cv2.IMREAD_GRAYSCALE
                )
                img = cv2.resize(img, (224, 224))
                img = img.astype(np.float32) / 255.0

                # 🔥 Convert to RGB
                img = cv2.cvtColor(
                    (img * 255).astype(np.uint8),
                    cv2.COLOR_GRAY2RGB
                )
                img = img.astype(np.float32) / 255.0

                suspect_images.append(img)

            suspect_images = np.array(suspect_images)
            st.success("✅ All suspect images uploaded")

# -------------------------------------------------
# Processing & Ranking
# -------------------------------------------------
if st.button("🔬 Run Identification"):

    if crime_img is None:
        st.error("Please upload a crime lip print.")
    elif suspect_images is None:
        st.error("Please upload all suspect images.")
    else:
        with st.spinner("Computing similarity rankings..."):

            crime_emb = encoder.predict(crime_img)
            suspect_embs = encoder.predict(suspect_images)

            distances = np.linalg.norm(
                suspect_embs - crime_emb,
                axis=1
            )

            ranked_indices = np.argsort(distances)

        st.subheader("📊 Ranked Suspect Matches")

        for rank, idx in enumerate(ranked_indices, 1):
            st.markdown(
                f"**Rank {rank}** | Similarity Distance: `{distances[idx]:.4f}`"
            )
            st.image(
                suspect_images[idx],
                width=150
            )
        st.success("✅ Identification completed")