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

from target_verifier import verify_targets

cap = cv2.VideoCapture(1)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    targets = verify_targets(
        frame
    )

    print("\n===================")

    for t in targets:

        print(t)

        x1, y1, x2, y2 = t["bbox"]

        # Bounding Box
        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        # Center Point
        cv2.circle(
            frame,
            (
                t["center_x"],
                t["center_y"]
            ),
            5,
            (0, 0, 255),
            -1
        )

        # Label
        label = (
            f"{t['class']} "
            f"Y:{t['yolo_confidence']:.2f} "
            f"D:{t['similarity']:.2f}"
        )

        cv2.putText(
            frame,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

        # Verification Status
        status = (
            "VERIFIED"
            if t["verified"]
            else "REJECTED"
        )

        color = (
            (0, 255, 0)
            if t["verified"]
            else (0, 0, 255)
        )

        cv2.putText(
            frame,
            status,
            (x1, y2 + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2
        )

    cv2.imshow(
        "Target Verification",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()

cv2.destroyAllWindows()