import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# Configuration
model_path = "/Users/christiansalz/Desktop/SPoHF-Yolov8/InsectClassificationModel/insect_classifier.h5"
img_size = (224, 224)

# Load the model
model = tf.keras.models.load_model(model_path)

# Load class labels
data_dir = "/Users/christiansalz/Desktop/SPoHF-Yolov8/Insect-Types-Classes"
class_names = sorted([name for name in os.listdir(data_dir) if not name.startswith('.')])  # Assumes class names are folder names
print(f"Loaded classes: {class_names}")

# Function to preprocess and predict a single image
def predict_image(img_path):
    # Load and preprocess the image
    img = image.load_img(img_path, target_size=img_size)
    img_array = image.img_to_array(img) / 255.0  # Rescale pixel values to [0, 1]
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Make a prediction
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)[0]
    confidence = np.max(predictions)

    return class_names[predicted_class], confidence

# Example usage
img_path = "/Users/christiansalz/Desktop/4.png"  # Replace with the path to your test image
if os.path.exists(img_path):
    predicted_class, confidence = predict_image(img_path)
    print(f"Predicted class: {predicted_class} (Confidence: {confidence:.2f})")
else:
    print(f"Image not found: {img_path}")
