from ultralytics import YOLO
import torch

# Check if MPS is available
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
#print(f"Training on device: {device}")

# Define model and dataset paths
model_path = "yolov8n.pt"  # Path to your model configuration
data_path = '/Users/christiansalz/Desktop/SPoHF-Yolov8/data.yaml'  # Path to your dataset YAML file

# Load the YOLO model
model = YOLO(model_path)

# Train the model
model.train(data=data_path, epochs=100, device=device, imgsz=640)  # Train the model on MPS or CPU