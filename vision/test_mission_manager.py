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
from mission_manager import process_verified_targets

cap = cv2.VideoCapture(1)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # =================================
    # YOLO + DINO Verification
    # =================================

    targets = verify_targets(
        frame
    )

    # =================================
    # Mission Manager
    # =================================

    found_targets = process_verified_targets(
        frame,
        targets
    )

    # =================================
    # Display Targets
    # =================================

    for t in targets:

        x1, y1, x2, y2 = t["bbox"]

        color = (
            (0, 255, 0)
            if t["verified"]
            else (0, 0, 255)
        )

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            color,
            2
        )

        cv2.putText(
            frame,
            t["class"],
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )

    cv2.imshow(
        "Mission Manager Test",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()

cv2.destroyAllWindows()