import roboflow

# Initialize Roboflow with your API key
rf = roboflow.Roboflow(api_key='YOUR-API-KEY')

# Load the project
project = rf.workspace().project("spohf-kur4x")

# Specify the dataset version ID
version = project.version('18')  # Replace VERSION_ID with the correct version ID

# Deploy the model weights
version.deploy("yolov8", "/Users/christiansalz/runs/detect/train10/weights", "last.pt")
