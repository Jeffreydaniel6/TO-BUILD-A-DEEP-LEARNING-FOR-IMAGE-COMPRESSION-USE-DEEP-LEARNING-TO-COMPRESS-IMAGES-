import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from build_model import build_compression_model

# Prepare data
datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    validation_split=0.2
)

train_data = datagen.flow_from_directory(
    'images/',
    target_size=(128, 128),
    batch_size=32,
    class_mode='input',
    subset='training'
)

val_data = datagen.flow_from_directory(
    'images/',
    target_size=(128, 128),
    batch_size=32,
    class_mode='input',
    subset='validation'
)

# Build and compile model
model = build_compression_model()
model.compile(optimizer='adam', loss='mean_absolute_error')

# Train model
print("Starting training for 50 epochs...")
history = model.fit(
    train_data,
    epochs=500,



    validation_data=val_data
)

# Save model
model.save('image_compression_model.h5')
print("Training complete. Model saved as 'image_compression_model.h5'.")
