import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import gdown

# ==============================
# MODEL DOWNLOAD CONFIG
# ==============================

FILE_ID = "1-53DJaGGe4r53HI7P-Y-WQ3lKNqQVlYM"
MODEL_URL = f"https://drive.google.com/uc?id=1-53DJaGGe4r53HI7P-Y-WQ3lKNqQVlYM"
MODEL_PATH = "fighting_movement_model.h5"

# ==============================
# LOAD MODEL (DOWNLOAD IF NEEDED)
# ==============================

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.write("Downloading model... Please wait ⏳")
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

    model = tf.keras.models.load_model(MODEL_PATH)
    return model

model = load_model()

# ==============================
# CLASS LABELS
# ==============================

class_indices = {'punch': 0, 'kick': 1, 'block': 2}
class_labels = {v: k for k, v in class_indices.items()}

# ==============================
# PREDICTION FUNCTION
# ==============================

def predict_fighting_movement(img):
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])

    return class_labels[predicted_class]

# ==============================
# UI DESIGN
# ==============================

st.markdown("""
<style>
.stApp {
    background-color: #ADD8E6;
}
</style>
""", unsafe_allow_html=True)

st.image("A-removebg-preview.png", width=400)
st.title(" Fighting Movement Predictor")

uploaded_file = st.file_uploader("Upload Fighting Image", type=["jpg", "png"])

if uploaded_file is not None:
    img = image.load_img(uploaded_file, target_size=(150, 150))
    st.image(img, caption='Uploaded Image', use_column_width=True)

    movement = predict_fighting_movement(img)
    st.success(f"Predicted Fighting Movement: {movement}")
