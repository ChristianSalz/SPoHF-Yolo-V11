from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import numpy as np
import os
import cv2
from ultralytics import YOLO
import io
import base64

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load model once at startup
model = YOLO('./model/last.pt')

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'}), 200

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check if image is in request
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        file = request.files['image']
        
        # Read image
        image = Image.open(file.stream)
        
        # Get parameters from request (optional)
        conf = float(request.form.get('conf', 0.40))
        iou = float(request.form.get('iou', 0.20))
        
        # Run inference
        results = model.predict(image, conf=conf, iou=iou)
        
        # Get detections
        detected_insects = results[0].boxes
        num_insects = len(detected_insects)
        
        # Prepare response data
        detections = []
        for box in detected_insects:
            detections.append({
                'class': int(box.cls[0]),
                'confidence': float(box.conf[0]),
                'bbox': box.xyxy[0].tolist()
            })
        
        # Optionally return annotated image
        return_image = request.form.get('return_image', 'false').lower() == 'true'
        
        response = {
            'num_insects': num_insects,
            'detections': detections
        }
        
        if return_image:
            # Generate annotated image
            annotated = results[0].plot(labels=True, font_size=6, line_width=2)
            annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            annotated_pil = Image.fromarray(annotated_rgb)
            
            # Convert to base64
            buffered = io.BytesIO()
            annotated_pil.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            response['annotated_image'] = img_str
        
        return jsonify(response), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 7733))
    app.run(host="0.0.0.0", port=port)
