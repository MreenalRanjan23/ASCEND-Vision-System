import sys
import os

sys.path.append(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

import cv2

from detector_node import process_frame

cap = cv2.VideoCapture(1)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    detections = process_frame(frame)

    print("\nDetections:")

    for d in detections:

        print(d)

    cv2.imshow(
        "Camera",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()