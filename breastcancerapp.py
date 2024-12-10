# -*- coding: utf-8 -*-
"""BreastCancerApp.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1nHJ1tIRqKZ7-kjJVdcxQPr8lOAzeDZ8G
"""

!touch cancerapp.py

from tensorflow.keras.models import load_model
model = load_model('/content/breast_cancer_model.keras')
model.summary()

!pip install streamlit

import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2
import time
import os
from tensorflow.keras.models import load_model

#Define preprocessing function
def preprocess_image(image, target_size):
  image = image.resize(target_size)
  image = np.array(image)
  image = np.expand_dims(image, axis=0)
  image =image / 255.0
  return image

  for percent_complete in range(101):
    time.sleep(0.1)
    progress_bar.progress(percent_complete)

st.title("Histopathological Breast Cancer Detection")
st.write("Upload a histopathological Image to detect if it is Benign or Malignant.")


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


# Uploading images
uploaded_file = st.file_uploader("Choose an Image", type=["tif"])

if uploaded_file is not None:
    #displays
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Initialize progress bar
    for percent_complete in range(101):
     time.sleep(1)
     progress_bar.progress(percent_complete)

    # # Convert the image to a numpy array
    # image = np.array(image)

    # # Ensure the image is RGB (some histopathology images might have 4 channels)
    # if image.shape[-1] == 4:
    #     image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    # preprocessing and prediction
    processed_image = preprocess_image(image, (224,224))
    prediction = model.predict(processed_image)
    result = "Malignant" if prediction[0][0] > 0.6 else "Benign"


    # Display results
    st.subheader("Prediction Results")
    class_names = ["Benign", "Malignant"]
    predicted_class = np.argmax(result, axis=1)[0]
    confidence = np.max(prediction) * 100
    st.write(f"**Prediction:** **{class_names[predicted_class]}**")
    st.write(f"**Confidence:** **{confidence:.2f}%**")

pip freeze > requirements.txt

!streamlit run cancerapp.py