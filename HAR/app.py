import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image
import joblib

IMAGE_SIZE = (224, 224)
mobilenet = MobileNetV2(weights='imagenet', include_top=False, pooling='avg')

def extract_features_from_image(img):
    img = img.resize(IMAGE_SIZE)
    img_array = image.img_to_array(img)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    features = mobilenet.predict(img_array, verbose=0)
    return features.flatten().reshape(1, -1)

@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

clf = load_model()

st.title("🤸 Human Activity Recognition (HAR) by image")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded image", use_column_width=True)

    st.write("🔍 Extracting features and making predictions...")
    features = extract_features_from_image(img)
    prediction = clf.predict(features)[0]

    st.success(f"🧠 Predicted activity: **{prediction}**")