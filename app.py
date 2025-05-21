import streamlit as st
import numpy as np
import cv2
from PIL import Image
import os
import matplotlib.pyplot as plt

# Define image dimensions
IMG_HEIGHT, IMG_WIDTH = 256, 256

# Function to preprocess images
def preprocess_image(image):
    image = image.resize((IMG_WIDTH, IMG_HEIGHT))
    image = np.array(image) / 255.0  # Normalize
    return image

# Function to preprocess masks (for visualization)
def preprocess_mask(mask):
    mask = mask.resize((IMG_WIDTH, IMG_HEIGHT))
    mask = np.array(mask) / 255.0  # Normalize
    mask = np.where(mask > 0.5, 1, 0)  # Convert to binary mask
    return mask

# Function to calculate wound percentage
def calculate_wound_percentage(mask):
    total_pixels = mask.shape[0] * mask.shape[1]
    wound_pixels = np.sum(mask)
    return (wound_pixels / total_pixels) * 100

# Function to predict healing status
def predict_healing_status(wound_percentage):
    if wound_percentage > 30:
        return "🟥 Severe Wound - Requires Medical Attention"
    elif 10 < wound_percentage <= 30:
        return "🟧 Moderate Wound - Healing in Progress"
    else:
        return "🟩 Minor Wound - Healing Well"

# Streamlit UI
st.title("🔍 Wound Segmentation & Healing Prediction")

st.write("Upload a wound image to analyze its segmented region and healing status.")

uploaded_file = st.file_uploader("Upload Wound Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Load and preprocess the image
    image = Image.open(uploaded_file).convert("RGB")
    processed_image = preprocess_image(image)

    # Dummy segmentation mask (Replace with ML model later)
    mask = np.random.randint(0, 2, (IMG_HEIGHT, IMG_WIDTH), dtype=np.uint8)  # Simulated segmentation mask
    processed_mask = preprocess_mask(Image.fromarray(mask * 255))  # Convert mask for visualization

    # Calculate wound percentage
    wound_percentage = calculate_wound_percentage(processed_mask)
    healing_status = predict_healing_status(wound_percentage)

    # Display results
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(image, caption="📸 Uploaded Image", use_column_width=True)
    
    with col2:
        st.image(processed_mask, caption="🔬 Segmented Wound Mask", use_column_width=True)
    
    st.subheader(f"Healing Status: {healing_status}")
    st.write(f"🩹 **Wound Coverage:** {wound_percentage:.2f}%")

st.write("📌 This is a prototype app. Model integration can be added later.")
