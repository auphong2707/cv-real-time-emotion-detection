import cv2
import numpy as np
import os

# [DOWNLOAD CAFFE MODEL FILES]
# Model: https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel
# Prototxt: https://github.com/opencv/opencv/blob/master/samples/dnn/face_detector/deploy.prototxt
os.makedirs("tmp", exist_ok=True)
# Download the Caffe model files
prototxt_url = "https://raw.githubusercontent.com/opencv/opencv/refs/heads/master/samples/dnn/face_detector/deploy.prototxt"
model_url = "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"

os.system("curl -o tmp/deploy.prototxt {}".format(prototxt_url))
os.system("curl -L -o tmp/res10_300x300_ssd_iter_140000.caffemodel {}".format(model_url))


# [FACE DETECTION USING DNN]
# Load the Caffe model files (update paths as needed)
prototxt_path = "./tmp/deploy.prototxt"
model_path = "./tmp/res10_300x300_ssd_iter_140000.caffemodel"

# Initialize the DNN
net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

# Start the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip frame horizontally (optional)
    frame = cv2.flip(frame, 1)

    # Prepare the image for the DNN: 
    #   - scale factor=1.0
    #   - target size=(300,300)
    #   - mean=(104.0, 177.0, 123.0)
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

            # Draw the bounding box of the face
            cv2.rectangle(
                frame, 
                (startX, startY), 
                (endX, endY),
                (255, 0, 0), 
                2
            )

            # Put a text label with confidence (as percentage)
            text = f"{confidence * 100:.2f}%"
            y_text = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.putText(
                frame, 
                text, 
                (startX, y_text), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.45, 
                (255, 0, 0), 
                2
            )

    # Show the frame
    cv2.imshow("DNN Face Detection", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# [CLEANUP]
# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()
