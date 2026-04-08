from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2
from ultralytics import YOLO
import tensorflow as tf
from PIL import ImageFont
import os

# Load the image
image_path = './Manual-Test-Data/5.jpg'
image = Image.open(image_path)

# Convert to RGB if it has alpha channel (PNG)
if image.mode in ('RGBA', 'LA', 'P'):
    image = image.convert('RGB')

image_np = np.array(image)

# Load the YOLO detection model
yolo_model = YOLO('./runs/detect/train9/weights/last.pt')

# Load the Keras classification model
classifier_model = tf.keras.models.load_model('./InsectClassificationModel/insect_classifier.keras')

# Get prediction parameters from environment variables with defaults
CONFIDENCE_THRESHOLD = float(os.getenv('CONFIDENCE_THRESHOLD', '0.70'))
IOU_THRESHOLD = float(os.getenv('IOU_THRESHOLD', '0.20'))

# Run YOLO inference to detect insects
results = yolo_model.predict(image, conf=CONFIDENCE_THRESHOLD, iou=IOU_THRESHOLD)

detected_insects = results[0].boxes
num_insects = len(detected_insects)

print(f"Number of insects detected: {num_insects}")

# Prepare image for drawing
draw_image = image.copy()
draw = ImageDraw.Draw(draw_image)

# Define colors for different classes
COLORS = {
    'Muscidae': (255, 0, 0),      # Red
    'Others': (0, 200, 0)          # Green
}

# Initialize counters
class_counts = {
    'Muscidae': 0,
    'Others': 0
}

# Load a font
font_paths = [
    "/System/Library/Fonts/Helvetica.ttc",  # macOS
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # Linux (Debian/Ubuntu)
    "/usr/share/fonts/dejavu/DejaVuSans.ttf",  # Linux (Fedora/RHEL)
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",  # Linux
    "C:\\Windows\\Fonts\\arial.ttf",  # Windows
]

font = None
for path in font_paths:
    try:
        font = ImageFont.truetype(path, 20)
        break
    except:
        continue

if font is None:
    # Fallback: use default font (will be smaller than requested)
    font = ImageFont.load_default()

# Process each detected insect
for idx, box in enumerate(detected_insects):
    # Get bounding box coordinates
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    
    # Crop the insect
    cropped_insect = image.crop((x1, y1, x2, y2))
    
    # Resize to model input size (224x224)
    cropped_resized = cropped_insect.resize((224, 224))
    
    # Convert to RGB in case crop has alpha channel
    if cropped_resized.mode != 'RGB':
        cropped_resized = cropped_resized.convert('RGB')
    
    # Prepare image for classification
    img_array = np.array(cropped_resized) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    
    # Predict with classifier
    prediction = classifier_model.predict(img_array, verbose=0)[0][0]
    
    # Determine class based on prediction
    if prediction < 0.5:
        class_name = 'Muscidae'
        confidence = (1 - prediction) * 100
    else:
        class_name = 'Others'
        confidence = prediction * 100
    
    # Increment counter for this class
    class_counts[class_name] += 1
    
    # Get color for this class
    color = COLORS[class_name]
    
    # Draw bounding box
    draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
    
    # Draw label with background
    label = f"{class_name} {confidence:.1f}%"
    
    # Get text bounding box
    bbox = draw.textbbox((x1, y1), label, font=font)
    
    # Draw background rectangle for text
    draw.rectangle([bbox[0]-2, bbox[1]-2, bbox[2]+2, bbox[3]+2], fill=color)
    
    # Draw text
    draw.text((x1, y1), label, fill=(255, 255, 255), font=font)
    
    print(f"Insect {idx+1}: {class_name} ({confidence:.1f}%)")

# Show the final annotated image
draw_image.show()

print(f"\nSummary:")
print(f"Total insects detected: {num_insects}")
print(f"Muscidae: {class_counts['Muscidae']}")
print(f"Others: {class_counts['Others']}")