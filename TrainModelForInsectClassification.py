import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
import os

# Configuration
data_dir = "./Insect-Types-Classes"
img_size = (224, 224)
batch_size = 1
epochs = 20  # Set higher, early stopping will control it
model_path = "./InsectClassificationModel/insect_classifier.keras"

# Data preparation
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
train_data = train_datagen.flow_from_directory(data_dir, target_size=img_size, batch_size=batch_size, subset="training", class_mode='binary')
val_data = train_datagen.flow_from_directory(data_dir, target_size=img_size, batch_size=batch_size, subset="validation", class_mode='binary')

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
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Print class indices to verify folder mapping
print("Class indices:", train_data.class_indices)
print(f"Total training samples: {train_data.samples}")
print(f"Total validation samples: {val_data.samples}")

# Early stopping callback
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True,
    verbose=1
)

# Train the model
history = model.fit(train_data, validation_data=val_data, epochs=epochs, callbacks=[early_stopping])

# Save the model
model.save(model_path)
print(f"Model saved to {model_path}")

# Print final accuracy
print(f"Final training accuracy: {history.history['accuracy'][-1]:.4f}")
print(f"Final validation accuracy: {history.history['val_accuracy'][-1]:.4f}")