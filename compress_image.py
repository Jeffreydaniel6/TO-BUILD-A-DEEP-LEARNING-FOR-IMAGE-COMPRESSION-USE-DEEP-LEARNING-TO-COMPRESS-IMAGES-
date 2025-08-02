import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array, save_img
import numpy as np
import os

# Load the model
model = tf.keras.models.load_model('image_compression_model.h5')

# Directory containing images
image_directory = 'images/'

# Loop through all images in the directory
for filename in os.listdir(image_directory):
    if filename.endswith(('.jpeg', '.jpg', '.png')):
        image_path = os.path.join(image_directory, filename)
        img = load_img(image_path, target_size=(128, 128))
        img_array = img_to_array(img) / 255.0  # Normalize the image
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        # Compress the image
        compressed_image = model.predict(img_array)

        # Save the compressed image as JPEG
        compressed_image = np.squeeze(compressed_image)  # Remove batch dimension
        compressed_image = (compressed_image * 255).astype(np.uint8)  # Convert back to uint8
        compressed_image_path = os.path.join(image_directory, f'compressed_{filename}.jpeg')
        save_img(compressed_image_path, compressed_image)  # Save the compressed image as JPEG
        print(f'Compressed image saved as {compressed_image_path}')
