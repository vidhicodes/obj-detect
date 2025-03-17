import streamlit as st
from PIL import Image
import numpy as np
import torch
from ultralytics import YOLO

model = YOLO("best.pt")  # ensure best.pt in same folder

st.title("Object Detection Web App")
st.write("Upload an image, and the model will detect objects.")

uploaded_image = st.file_uploader("Upload an image", type=["jpg", "png"])

if uploaded_image:
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    image_np = np.array(image)
    results = model(image_np)

    st.image(results[0].plot(), caption="Detected Objects")

    st.write("Predictions:")
    for box in results[0].boxes:
        class_id = int(box.cls[0])
        confidence = round(float(box.conf[0]), 2)
        st.write(f"Class: {model.names[class_id]} | Confidence: {confidence}")
