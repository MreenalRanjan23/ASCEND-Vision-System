from ultralytics import YOLO
import os

# =====================================
# Model Path
# =====================================

MODEL_PATH = os.path.join(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    ),
    "models",
    "best.pt"
)

# =====================================
# Load Model Once
# =====================================

model = YOLO(MODEL_PATH)

print("YOLO detector loaded")

# =====================================
# Detection Function
# =====================================

def detect(frame):

    results = model(
        frame,
        conf=0.25,
        imgsz=1280,
        verbose=False
    )

    detections = []

    if len(results[0].boxes) == 0:
        return detections

    for box in results[0].boxes:

        cls_id = int(box.cls[0])

        confidence = float(
            box.conf[0]
        )

        x1, y1, x2, y2 = map(
            int,
            box.xyxy[0]
        )

        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2

        detections.append({

            "class":
                model.names[cls_id],

            "confidence":
                confidence,

            "bbox":
                [x1, y1, x2, y2],

            "center":
                [center_x, center_y]
        })

    return detections