import argparse
import cv2
import numpy as np
import os
import torch
import torchvision.transforms as transforms
import sys
import asyncio
from fastapi import FastAPI, WebSocket, Request
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
import logging
from PIL import Image
import io
import time

# === BEGIN: CLI args for device selection ===
parser = argparse.ArgumentParser(description="Face-emotion FastAPI server")
parser.add_argument(
    '--cpu',
    action='store_true',
    help='Force the model to run on CPU even if GPU is available'
)
args, _ = parser.parse_known_args()

# Thiết lập device
if args.cpu:
    device = torch.device("cpu")
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logging.info(f"Using device: {device}")

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Add the models directory to the system path
sys.path.append("models")

# Import model functions
from models.vgg16 import get_vgg16
from models.efficientnet_b0 import get_efficientnet_b0

# [CREATE DIRECTORIES]
os.makedirs("tmp", exist_ok=True)
os.makedirs("deploy_models", exist_ok=True)
os.makedirs("templates", exist_ok=True)

# [CHECK AND DOWNLOAD CAFFE MODEL FILES IF NEEDED]
prototxt_path = os.path.join("tmp", "deploy.prototxt")
model_path = os.path.join("tmp", "res10_300x300_ssd_iter_140000.caffemodel")
prototxt_url = "https://raw.githubusercontent.com/opencv/opencv/refs/heads/master/samples/dnn/face_detector/deploy.prototxt"
model_url = "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"

if not (os.path.exists(prototxt_path) and os.path.exists(model_path)):
    logging.info("Downloading Caffe model files...")
    os.system(f"curl -o {prototxt_path} {prototxt_url}")
    os.system(f"curl -L -o {model_path} {model_url}")
else:
    logging.info("Caffe model files already exist, skipping download.")

# [LIST MODELS IN deploy_models FOLDER]
model_files = [f for f in os.listdir("deploy_models") if f.endswith(".pth")]
if not model_files:
    raise FileNotFoundError("No .pth files found in deploy_models/ folder")

# [LOAD MODEL]
if args.cpu:
    device = torch.device("cpu")
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(model_path):
    model_name = os.path.basename(model_path).lower()
    if "vgg16" in model_name:
        logging.info(f"Loading VGG16 model: {model_name}")
        model = get_vgg16(num_classes=8, pretrained=False, freeze=False)
    elif "efficientnet" in model_name:
        logging.info(f"Loading EfficientNet-B0 model: {model_name}")
        model = get_efficientnet_b0(num_classes=8, pretrained=False, freeze=False)
    elif "mobilenetv2" in model_name:
        logging.info(f"Loading MobileNetV2 model: {model_name}")
        from models.mobilenetv2 import get_mobilenetv2
        model = get_mobilenetv2(num_classes=8, pretrained=False, freeze=False)
    elif "emotioncnn" in model_name:
        logging.info(f"Loading Custom model: {model_name}")
        from models.emotion_cnn import EmotionCNN
        model = EmotionCNN(num_classes=8)
    else:
        logging.warning(f"Unknown model type for {model_name}, defaulting to VGG16")
        model = get_vgg16(num_classes=8, pretrained=False, freeze=False)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model

# Initialize with the first model
current_model_idx = 0
current_model_path = os.path.join("deploy_models", model_files[current_model_idx])
model = load_model(current_model_path)

# Define preprocessing
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Define class labels
class_labels = ["Anger", "Contempt", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

# [FACE DETECTION USING DNN]
net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

# Initialize webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# FastAPI app
app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Global variables
current_model_idx = 0
model = load_model(os.path.join("deploy_models", model_files[current_model_idx]))
fps = 0
last_time = time.time()

def generate_frames():
    global fps, last_time
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        # Flip frame
        frame = cv2.flip(frame, 1)
        (h, w) = frame.shape[:2]

        # Face detection
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
        net.setInput(blob)
        detections = net.forward()

        # List to collect all faces' probabilities
        faces_data = []

        # Process all detected faces
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype(int)

                # Expand bounding box
                box_width = endX - startX
                box_height = endY - startY
                margin_top = int(box_height * 0.3)
                margin_side = int(box_width * 0.1)
                startX = max(0, startX - margin_side)
                startY = max(0, startY - margin_top)
                endX = min(w - 1, endX + margin_side)
                endY = min(h - 1, endY)

                face = frame[startY:endY, startX:endX]
                if face.size == 0:
                    continue

                # Inference
                face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face_tensor = preprocess(face_rgb).unsqueeze(0).to(device)
                with torch.no_grad():
                    outputs = model(face_tensor)
                    probabilities = torch.softmax(outputs, dim=1)
                    # Collect probabilities for this face
                    probs = probabilities[0].cpu().numpy().tolist()
                    faces_data.append({"probabilities": probs})

                    # For drawing on the frame
                    predicted_class = torch.argmax(probabilities, dim=1).item()
                    class_prob = probabilities[0, predicted_class].item()

                predicted_label = class_labels[predicted_class]

                # Draw bounding box and text for each face
                cv2.rectangle(frame, (startX, startY), (endX, endY), (255, 0, 0), 2)
                text_conf = f"Conf: {confidence * 100:.2f}%"
                text_class = f"{predicted_label} ({class_prob * 100:.2f}%)"
                y_text_conf = startY - 30 if startY - 30 > 30 else startY + 30
                y_text_class = startY - 10 if startY - 10 > 10 else startY + 50
                cv2.putText(frame, text_conf, (startX, y_text_conf), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                cv2.putText(frame, text_class, (startX, y_text_class), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        # Update FPS
        current_time = time.time()
        fps = 1 / (current_time - last_time) if current_time != last_time else fps
        last_time = current_time

        # Encode frame as JPEG
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        # Send probabilities of all detected faces via WebSocket
        status = f"Model: {model_files[current_model_idx]} | FPS: {fps:.2f}"
        data = {"faces": faces_data, "status": status}
        asyncio.run(notify_clients(data))

# WebSocket clients
clients = set()

async def notify_clients(data):
    if clients:
        for client in clients:
            try:
                await client.send_json(data)
            except:
                clients.remove(client)

@app.get("/", response_class=HTMLResponse)
async def get_index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "model_files": model_files, "current_model": model_files[current_model_idx]})

@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    clients.add(websocket)
    try:
        while True:
            await websocket.receive_text()  # Keep connection alive
    except:
        clients.remove(websocket)

@app.post("/switch_model/{model_idx}")
async def switch_model_endpoint(model_idx: int):
    global current_model_idx, model
    if 0 <= model_idx < len(model_files):
        current_model_idx = model_idx
        current_model_path = os.path.join("deploy_models", model_files[current_model_idx])
        model = load_model(current_model_path)
        status = f"Switched to {model_files[current_model_idx]}"
        await notify_clients([0]*len(class_labels))
        return {"status": status}
    return {"error": "Invalid model index"}

@app.on_event("shutdown")
def cleanup():
    cap.release()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)