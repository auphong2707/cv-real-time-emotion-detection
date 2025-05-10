import cv2
import numpy as np
import os
import torch
import torchvision.transforms as transforms
import sys

# Add the models directory to the system path to import vgg16.py and efficientnet_b0.py
sys.path.append("models")

# Import the model functions
from vgg16 import get_vgg16
from efficientnet_b0 import get_efficientnet_b0

# [CREATE DIRECTORIES]
# Create tmp folder for face detection model files
os.makedirs("tmp", exist_ok=True)
# Create deploy_models folder for VGG16, EfficientNet-B0, and other model files
os.makedirs("deploy_models", exist_ok=True)

# [CHECK AND DOWNLOAD CAFFE MODEL FILES IF NEEDED]
# Paths to the Caffe model files
prototxt_path = os.path.join("tmp", "deploy.prototxt")
model_path = os.path.join("tmp", "res10_300x300_ssd_iter_140000.caffemodel")

# URLs for the Caffe model files
prototxt_url = "https://raw.githubusercontent.com/opencv/opencv/refs/heads/master/samples/dnn/face_detector/deploy.prototxt"
model_url = "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"

# Download the files only if they don't exist
if not (os.path.exists(prototxt_path) and os.path.exists(model_path)):
    print("Downloading Caffe model files...")
    os.system(f"curl -o {prototxt_path} {prototxt_url}")
    os.system(f"curl -L -o {model_path} {model_url}")
else:
    print("Caffe model files already exist, skipping download.")

# [LIST MODELS IN deploy_models FOLDER]
# Get a list of .pth files in the deploy_models folder
model_files = [f for f in os.listdir("deploy_models") if f.endswith(".pth")]
if not model_files:
    raise FileNotFoundError("No .pth files found in deploy_models/ folder")

# [LOAD MODEL]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Function to load a model based on its file name
def load_model(model_path):
    # Determine the model architecture based on the file name
    model_name = os.path.basename(model_path).lower()
    if "vgg16" in model_name:
        print(f"Loading VGG16 model: {model_name}")
        model = get_vgg16(num_classes=8, pretrained=False, freeze=False)
    elif "efficientnet" in model_name:
        print(f"Loading EfficientNet-B0 model: {model_name}")
        model = get_efficientnet_b0(num_classes=8, pretrained=False, freeze=False)
    else:
        # Default to VGG16 if the model type is unclear
        print(f"Unknown model type for {model_name}, defaulting to VGG16")
        model = get_vgg16(num_classes=8, pretrained=False, freeze=False)

    # Load the weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model

# Initialize with the first model
current_model_idx = 0
current_model_path = os.path.join("deploy_models", model_files[current_model_idx])
model = load_model(current_model_path)

# Define preprocessing for VGG16 and EfficientNet-B0 (224x224, normalized)
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Define class labels for 8 emotions based on ID2LABEL
class_labels = ["Anger", "Contempt", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

# [FACE DETECTION USING DNN]
# Load the Caffe model files
net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

# Start the webcam
cap = cv2.VideoCapture(0)

# Set camera to maximum resolution (try 1920x1080)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

# Get actual resolution
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Window name
window_name = "DNN Face Detection with Model Switching"

# Create the window and set size to match camera resolution
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, width, height)

# Define expansion factors for the bounding box
top_expansion_factor = 0.3  # 30% expansion on the top
side_expansion_factor = 0.1  # 10% expansion on left and right

# Define button properties
button_width = 200
button_height = 50
button_x = 10
button_y = 10
button_color = (0, 255, 0)  # Green button
button_text_color = (0, 0, 0)  # Black text

# Mouse callback function to handle button clicks
def switch_model(event, x, y, flags, param):
    global current_model_idx, model
    if event == cv2.EVENT_LBUTTONDOWN:
        # Check if the click is within the button area
        if button_x <= x <= button_x + button_width and button_y <= y <= button_y + button_height:
            # Cycle to the next model
            current_model_idx = (current_model_idx + 1) % len(model_files)
            current_model_path = os.path.join("deploy_models", model_files[current_model_idx])
            model = load_model(current_model_path)

# Set the mouse callback for the window
cv2.setMouseCallback(window_name, switch_model)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip frame horizontally (optional)
    frame = cv2.flip(frame, 1)

    # Prepare the image for the DNN
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(
        frame, 
        scalefactor=1.0, 
        size=(300, 300), 
        mean=(104.0, 177.0, 123.0)
    )

    # Set the input blob for the network
    net.setInput(blob)

    # Forward pass to get face detections
    detections = net.forward()

    # Loop over each detection
    for i in range(0, detections.shape[2]):
        # Confidence/probability associated with the prediction
        confidence = detections[0, 0, i, 2]

        # Filter out weak detections (e.g., confidence < 0.5)
        if confidence > 0.5:
            # The bounding box is given by (startX, startY, endX, endY)
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype(int)

            # Calculate the width and height of the bounding box
            box_width = endX - startX
            box_height = endY - startY

            # Expand the bounding box asymmetrically
            margin_top = int(box_height * top_expansion_factor)  # Expand top by 30%
            margin_side = int(box_width * side_expansion_factor)  # Expand left/right by 10%

            # Apply the margins to the coordinates
            startX = max(0, startX - margin_side)  # Expand left
            startY = max(0, startY - margin_top)   # Expand top
            endX = min(w - 1, endX + margin_side)  # Expand right
            endY = min(h - 1, endY)                # No expansion on bottom

            # Extract the expanded face ROI
            face = frame[startY:endY, startX:endX]
            if face.size == 0:  # Skip if the ROI is empty
                continue

            # Preprocess the face for the model
            face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
            face_tensor = preprocess(face_rgb).unsqueeze(0).to(device)  # Add batch dimension

            # Perform inference with the current model
            with torch.no_grad():
                outputs = model(face_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).item()
                class_prob = probabilities[0, predicted_class].item()

            # Get the predicted class label
            predicted_label = class_labels[predicted_class]

            # Draw the expanded bounding box of the face
            cv2.rectangle(
                frame, 
                (startX, startY), 
                (endX, endY),
                (255, 0, 0), 
                2
            )

            # Put text labels: confidence and predicted class
            text_conf = f"Conf: {confidence * 100:.2f}%"
            text_class = f"Class: {predicted_label} ({class_prob * 100:.2f}%)"
            y_text_conf = startY - 30 if startY - 30 > 30 else startY + 30
            y_text_class = startY - 10 if startY - 10 > 10 else startY + 50
            cv2.putText(
                frame, 
                text_conf, 
                (startX, y_text_conf), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.45, 
                (255, 0, 0), 
                2
            )
            cv2.putText(
                frame, 
                text_class, 
                (startX, y_text_class), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.45, 
                (255, 0, 0), 
                2
            )

    # Draw the "Switch Model" button
    cv2.rectangle(
        frame,
        (button_x, button_y),
        (button_x + button_width, button_y + button_height),
        button_color,
        -1  # Filled rectangle
    )

    # Add text to the button
    button_text = "Switch Model"
    cv2.putText(
        frame,
        button_text,
        (button_x + 10, button_y + 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        button_text_color,
        2
    )

    # Display the current model name below the button
    current_model_name = model_files[current_model_idx]
    cv2.putText(
        frame,
        f"Model: {current_model_name}",
        (button_x, button_y + button_height + 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),  # White text
        2
    )

    # Show the frame
    cv2.imshow(window_name, frame)

    # Check for 'q' key or window close event
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
        break

# [CLEANUP]
# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()