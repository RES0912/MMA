import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2

# Load Model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("mma_movement_model.h5")
    return model

model = load_model()

class_names = ['Punch', 'Kick', 'Block', 'Elbow', 'Knee']

st.set_page_config(page_title="MMA Movement Predictor")

st.title("🥋 MMA Movement Prediction")

uploaded_file = st.file_uploader("Upload Fighter Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image")

    img = np.array(image)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction)

    st.success(f"Predicted Movement: {predicted_class}")
    st.info(f"Confidence: {confidence:.2f}")
