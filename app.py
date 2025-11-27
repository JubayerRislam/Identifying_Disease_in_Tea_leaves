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
    .stButton>button {
        background-color: #26a69a;
        color: white;
        font-size: 16px;
        border-radius: 8px;
    }
    .stFileUploader>div>div>label {
        font-weight: bold;
        color: #00796b;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🍃 Tea Leaf Disease Classifier")
st.write(
    "Upload an image of a tea leaf and the model will predict which disease (if any) it has."
)

# ---- LOAD MODEL ----
@st.cache_resource
def load_tea_model():
    return load_model("tea_leaf_model.h5")  # replace with your saved .h5 path

model = load_tea_model()

# ---- IMAGE UPLOAD ----
uploaded_file = st.file_uploader(
    "Choose a tea leaf image...",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    # Load image
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    IMG_SIZE = (224, 224)  # match your model input
    img_resized = img.resize(IMG_SIZE)
    img_array = image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # normalize

    # Prediction
    pred_probs = model.predict(img_array)[0]
    pred_idx = np.argmax(pred_probs)

    # Class labels (must match your training class_indices)
    class_labels = sorted([
        'Algal leaf spot', 'Bird’s eye spot', 'Grey blight', 'Red rust', 'Tea mosquito bug'
    ])
    predicted_label = class_labels[pred_idx]
    confidence = pred_probs[pred_idx] * 100

    # Display prediction
    st.markdown(
        f"### 🍀 Predicted Disease: **{predicted_label}**"
    )
    st.markdown(
        f"**Confidence:** {confidence:.2f}%"
    )

    # Optional: show top 3 predictions
    top3_idx = pred_probs.argsort()[-3:][::-1]
    st.markdown("#### 🔹 Top 3 Predictions:")
    for i in top3_idx:
        st.write(f"{class_labels[i]}: {pred_probs[i]*100:.2f}%")