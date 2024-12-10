## saving as app.py
import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2
import time

# Load the trained model
model = tf.keras.models.load_model("breast_cancer_model.keras")

# Define a function for making predictions
def predict_image(image):
    # Preprocess the image
    img_array = cv2.resize(image, (96, 96))  # Resize to model input size
    img_array = img_array / 255.0           # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Simulate a process to show the progress bar
    for percent_complete in range(101):
        time.sleep(0.01)  # Simulate computation time
        progress_bar.progress(percent_complete)

    # Make prediction
    prediction = model.predict(img_array)
    return prediction

# Streamlit App
st.title("Histopathological Breast Cancer Detection")
st.write("Upload a histopathology image to detect if it is benign or malignant.")

# Upload image
uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png", "tif"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Convert the image to a numpy array
    image = np.array(image)

    # Ensure the image is RGB (some histopathology images might have 4 channels)
    if image.shape[-1] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

    # Initialize progress bar
    st.subheader("Processing...")
    progress_bar = st.progress(0)

    # Make prediction
    with st.spinner("Making prediction..."):
        prediction = predict_image(image)

    # Display results
    st.subheader("Prediction Results")
    class_names = ["Benign", "Malignant"]
    predicted_class = np.argmax(prediction, axis=1)[0]
    confidence = np.max(prediction) * 100
    st.write(f"**Class:** {class_names[predicted_class]}")
    st.write(f"**Confidence:** {confidence:.2f}%")
