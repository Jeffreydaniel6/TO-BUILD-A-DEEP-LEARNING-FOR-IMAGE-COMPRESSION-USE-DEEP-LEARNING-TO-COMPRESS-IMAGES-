import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

def build_compression_model():
    inputs = layers.Input(shape=(128, 128, 3))

    # Encoder (Compression)
    conv1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    pool1 = layers.MaxPooling2D((2, 2), padding='same')(conv1)

    conv2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(pool1)
    pool2 = layers.MaxPooling2D((2, 2), padding='same')(conv2)

    conv3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(pool2)
    pool3 = layers.MaxPooling2D((2, 2), padding='same')(conv3)

    # Enhanced Bottleneck for better compression
    bottleneck = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(pool3)

    # Decoder (Decompression)
    upconv3 = layers.Conv2DTranspose(256, (3, 3), activation='relu', padding='same', strides=(2, 2))(bottleneck)
    upconv3 = layers.concatenate([upconv3, conv3])

    upconv2 = layers.Conv2DTranspose(128, (3, 3), activation='relu', padding='same', strides=(2, 2))(upconv3)
    upconv2 = layers.concatenate([upconv2, conv2])

    upconv1 = layers.Conv2DTranspose(64, (3, 3), activation='relu', padding='same', strides=(2, 2))(upconv2)
    upconv1 = layers.concatenate([upconv1, conv1])

    # Final reconstruction with residual connection
    outputs = layers.Conv2D(3, (1, 1), activation='sigmoid')(upconv1)

    model = models.Model(inputs, outputs)
    return model

# Build and compile model with improved settings
model = build_compression_model()
model.compile(optimizer='adam', loss='mean_absolute_error')

# Display model summary
model.summary()
