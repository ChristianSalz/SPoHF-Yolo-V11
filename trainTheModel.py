from ultralytics import YOLO
import torch

# Define model and dataset paths
model_path = "yolo11n.pt"  # You can use n, s, m , l  and XL depending on the needed complecity, a bigger modle needs more computational power
data_path = './data.yaml'  # Path to your dataset YAML file

# Load the YOLO model
model = YOLO(model_path)

# Train the model
# device="mps" runs on Apple metal you can alos use device="CPU" or if you have a cuda card device="CUDA" you can also use device="AUTO" if you dont know hich GPU you have CUDA > MPS > CPU
model.train(data=data_path, epochs=5, device="mps", imgsz=320)  