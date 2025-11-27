import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# ---- PAGE CONFIG ----
st.set_page_config(
    page_title="Tea Leaf Disease Classifier",
    page_icon="🍃",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ---- CUSTOM CSS ----
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(to right, #fdf6e3, #e0f7fa);
        color: #333333;
        font-family: 'Helvetica', sans-serif;
    }
    /* White Browse File button */
    div.stFileUploader > div > label > div[data-baseweb="file-input"] {
        background-color: white;
        color: #333;
        border-radius: 8px;
        padding: 8px 12px;
        font-weight: bold;
        border: 1px solid #ccc;
    }
    .stButton>button {
        background-color: #26a69a;
        color: white;
        font-size: 16px;
        border-radius: 8px;
    }
    .stProgress > div > div > div > div {
        background-color: #26a69a;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ---- APP TITLE ----
st.title("🍃 Tea Leaf Disease Classifier")
st.write(
    "Upload a tea leaf image, and the model will predict the disease along with confidence."
)

# ---- LOAD MODEL ----
@st.cache_resource
def load_tea_model():
    model = load_model("tea_leaf_model.h5")  # your saved model
    # Hardcode the class mapping exactly as learned during training
    inv_map = {
        0: "Anthracnose",
        1: "algal leaf",
        2: "bird eye spot",
        3: "brown blight",
        4: "gray light",
        5: "healthy",
        6: "red leaf spot",
        7: "white spot"
    }
    return model, inv_map

model, inv_map = load_tea_model()
IMG_SIZE = (224, 224)  # match your model input size

# ---- IMAGE UPLOAD ----
uploaded_file = st.file_uploader(
    "Choose a tea leaf image...",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    # Load and display image
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img_resized = img.resize(IMG_SIZE)
    img_array = image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # normalize

    # Predict
    pred_probs = model.predict(img_array)[0]
    pred_idx = np.argmax(pred_probs)
    predicted_label = inv_map[pred_idx]
    confidence = pred_probs[pred_idx] * 100

    # Display prediction
    st.markdown(f"### 🍀 Predicted Disease: **{predicted_label}**")
    st.markdown(f"**Confidence:** {confidence:.2f}%")

    # Show top-3 predictions
    st.markdown("#### 🔹 Top 3 Predictions:")
    top3_idx = pred_probs.argsort()[-3:][::-1]
    for i in top3_idx:
        label = inv_map[i]
        prob = pred_probs[i] * 100
        st.write(f"{label}: {prob:.2f}%")
        st.progress(int(prob))
