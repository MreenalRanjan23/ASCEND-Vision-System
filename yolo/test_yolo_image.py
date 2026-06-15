from ultralytics import YOLO
import cv2

# =====================================
# Load Model
# =====================================

model = YOLO("models/best.pt")

# =====================================
# Test Image
# =====================================

image_path = "test_images/test2.jpeg"

# =====================================
# Run Inference
# =====================================

results = model(
    image_path,
    conf=0.25,
    imgsz=1280
)

# =====================================
# Draw Predictions
# =====================================

annotated_frame = results[0].plot()

# =====================================
# Show Image
# =====================================

cv2.imshow(
    "YOLO Detection",
    annotated_frame
)

cv2.waitKey(0)
cv2.destroyAllWindows()

# =====================================
# Print Detections
# =====================================

print("\nDetections:\n")

for box in results[0].boxes:

    cls_id = int(box.cls[0])

    confidence = float(box.conf[0])

    print(
        f"Class: {model.names[cls_id]}"
    )

    print(
        f"Confidence: {confidence:.3f}"
    )

    print("-" * 30)