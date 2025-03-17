import streamlit as st
from PIL import Image
import torch
from ultralytics import YOLO
import numpy as np

model = YOLO("best.pt")  # Ensure best.pt is in the same folder

st.title("Object Detection Using YOLO")
st.write("Upload an image to detect objects")

uploaded_img = st.file_uploader("Upload an image", type=["jpg", "png"])

if uploaded_img:
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    image_np = np.array(image)

    results = model(image_np)

    st.image(results[0].plot(), caption="Detected Objects")  
