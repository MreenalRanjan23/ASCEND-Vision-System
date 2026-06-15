import sys
import os

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

import cv2

from yolo.detector import detect

cap = cv2.VideoCapture(1)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    detections = detect(frame)

    if len(detections) > 0:

        print("\n====================")

        for d in detections:

            print(
                f"{d['class']} | "
                f"Conf={d['confidence']:.2f} | "
                f"Center={d['center']}"
            )

    for d in detections:

        x1, y1, x2, y2 = d["bbox"]

        confidence = d["confidence"]

        center_x, center_y = d["center"]

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"{d['class']} {confidence:.2f}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

        cv2.circle(
            frame,
            (center_x, center_y),
            5,
            (0, 0, 255),
            -1
        )

    cv2.imshow(
        "YOLO Pipeline",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()

cv2.destroyAllWindows()