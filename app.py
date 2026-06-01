import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageEnhance

st.set_page_config(page_title="PCD Web Editor", layout="wide")

st.title("🎨 Digital Image Processing Editor")
st.write("Unggah foto dan terapkan filter secara real-time.")

# Sidebar untuk upload
uploaded_file = st.sidebar.file_uploader("Pilih gambar...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load gambar
    image = Image.open(uploaded_file)
    img_array = np.array(image.convert('RGB'))
    
    # Layout kolom: Kiri (Asli) | Kanan (Hasil)
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("Citra Asli")
        st.image(image, use_container_width=True)

    # Sidebar Kontrol
    st.sidebar.header("Pengaturan Filter")
    
    # 1. Image Enhancement
    brightness = st.sidebar.slider("Brightness", 0.5, 3.0, 1.0)
    contrast = st.sidebar.slider("Contrast", 0.5, 3.0, 1.0)
    
    # 2. Efek Spesifik
    filter_type = st.sidebar.selectbox("Pilih Filter", 
        ["None", "Grayscale", "Canny Edge Detection", "Gaussian Blur", "Sketch Effect", "Sepia"])

    # Proses Image Enhancement
    enhancer_b = ImageEnhance.Brightness(image)
    proc_img = enhancer_b.enhance(brightness)
    enhancer_c = ImageEnhance.Contrast(proc_img)
    proc_img = enhancer_c.enhance(contrast)
    
    # Konversi ke OpenCV format untuk filter
    open_cv_image = np.array(proc_img)
    open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)

    # Logika Filter
    if filter_type == "Grayscale":
        open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)
    
    elif filter_type == "Canny Edge Detection":
        low = st.sidebar.number_input("Canny Low Threshold", 0, 255, 100)
        high = st.sidebar.number_input("Canny High Threshold", 0, 255, 200)
        open_cv_image = cv2.Canny(open_cv_image, low, high)
        
    elif filter_type == "Gaussian Blur":
        k_size = st.sidebar.slider("Kernel Size", 1, 25, 5, step=2)
        open_cv_image = cv2.GaussianBlur(open_cv_image, (k_size, k_size), 0)
        
    elif filter_type == "Sketch Effect":
        gray = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)
        inv = cv2.bitwise_not(gray)
        blur = cv2.GaussianBlur(inv, (21, 21), 0)
        sketch = cv2.divide(gray, cv2.bitwise_not(blur), scale=256.0)
        open_cv_image = sketch

    elif filter_type == "Sepia":
        kernel = np.array([[0.272, 0.534, 0.131],
                           [0.349, 0.686, 0.168],
                           [0.393, 0.769, 0.189]])
        open_cv_image = cv2.transform(open_cv_image, kernel)

    with col2:
        st.header("Hasil Proses")
        st.image(open_cv_image, use_container_width=True, channels="BGR" if len(open_cv_image.shape) > 2 else "GRAY")
        
        # Fitur Download
        result_img = Image.fromarray(cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2RGB) if len(open_cv_image.shape) > 2 else open_cv_image)
        st.download_button(label="Download Hasil", data=uploaded_file, file_name="hasil_proses.png", mime="image/png")

else:
    st.info("Silakan unggah gambar di sidebar untuk memulai.")
    
