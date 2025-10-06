import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import os

# Configuration
data_dir = "/Users/christiansalz/Desktop/SPoHF-Yolov8/Insect-Types-Classes"
img_size = (224, 224)
batch_size = 1
epochs = 25
model_path = "/Users/christiansalz/Desktop/SPoHF-Yolov8/InsectClassificationModel/insect_classifier.h5"

# Data preparation
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
train_data = train_datagen.flow_from_directory(data_dir, target_size=img_size, batch_size=batch_size, subset="training")
val_data = train_datagen.flow_from_directory(data_dir, target_size=img_size, batch_size=batch_size, subset="validation")

# Build the model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(*img_size, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(train_data.num_classes, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_data, validation_data=val_data, epochs=epochs)

# Save the model
model.save(model_path)
print(f"Model saved to {model_path}")
