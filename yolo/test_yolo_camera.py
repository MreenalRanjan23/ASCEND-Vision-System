from ultralytics import YOLO
import cv2

# Load model
model = YOLO("models/best.pt")

# DroidCam index
cap = cv2.VideoCapture(1)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    results = model(
        frame,
        conf=0.25,
        imgsz=1280
    )

    annotated = results[0].plot()

    cv2.imshow(
        "YOLO Live Detection",
        annotated
    )

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()