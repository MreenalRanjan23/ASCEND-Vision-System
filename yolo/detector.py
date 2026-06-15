from ultralytics import YOLO

MODEL_PATH = "models/best.pt"

model = YOLO(MODEL_PATH)

def detect(frame):
    results = model(frame, conf=0.25)

    detections = []

    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            detections.append({
                "class_id": cls_id,
                "class_name": model.names[cls_id],
                "confidence": conf,
                "bbox": [x1, y1, x2, y2]
            })

    return detections