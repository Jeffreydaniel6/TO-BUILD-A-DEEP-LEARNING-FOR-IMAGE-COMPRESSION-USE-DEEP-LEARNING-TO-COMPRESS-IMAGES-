from io import BytesIO
import streamlit as st

from PIL import Image
import numpy as np
import tensorflow as tf

# Load the model
model = tf.keras.models.load_model('image_compression_model.h5')  # Load the newly trained model


def compress_image(image):
    image = image.resize((128, 128))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    compressed_image = model.predict(image_array)
    compressed_image = np.clip(compressed_image, 0, 1)  # Ensure values are within valid range
    return np.squeeze(compressed_image)

st.title("Image Compression with Deep Learning")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    if st.button("Compress Image"):
        compressed_image = compress_image(image)
        st.image(compressed_image, caption='Compressed Image', use_column_width=True)
        # Convert the compressed image to a format suitable for download
        compressed_image_pil = Image.fromarray((compressed_image * 255).astype(np.uint8))
        buffered = BytesIO()
        compressed_image_pil.save(buffered, format="JPEG")  # Save as JPEG
        compressed_image_bytes = buffered.getvalue()
        
        st.download_button("Download Compressed Image", compressed_image_bytes, "compressed_image.jpeg", "image/jpeg")  # Change to JPEG
