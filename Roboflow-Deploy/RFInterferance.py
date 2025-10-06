import cv2
from inference_sdk import InferenceHTTPClient

# Paths
image_path = "/Users/christiansalz/Desktop/SPoHF-Yolov8/Study-Data/18.jpg"
#resized_image_path = "/Users/christiansalz/Desktop/SPoHF-Yolov8/Study-Data/16_resized.jpg"
annotated_image_path = "/Users/christiansalz/Desktop/SPoHF-Yolov8/Study-Data/18_annotated.jpg"

# Load and resize the image
image = cv2.imread(image_path)
#resized_image = cv2.resize(image, (4096, 4096))
#cv2.imwrite(resized_image_path, resized_image)

# Create an inference client
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="FtYecAJ0Re1CyfvcGd2A"  # Ensure this is a valid API key
)

# Run inference on the resized image
response_json = CLIENT.infer(image, model_id="spohf-kur4x-dokg9-rxwfv/1")

# Draw bounding boxes
for prediction in response_json.get("predictions", []):
    x, y, w, h = int(prediction["x"]), int(prediction["y"]), int(prediction["width"]), int(prediction["height"])
    confidence = prediction["confidence"]

    # Convert center (x, y) to top-left corner
    x1, y1 = x - w // 2, y - h // 2
    x2, y2 = x + w // 2, y + h // 2

    # Draw rectangle
    cv2.rectangle(image, (x1, y1), (x2, y2), (252, 3, 211), 2)

    # Add label with confidence
    label = f"{prediction['class']} {confidence:.2f}"
    cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (252, 3, 211), 2)

# Save and display annotated image
cv2.imwrite(annotated_image_path, image)
cv2.imshow("Annotated Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()