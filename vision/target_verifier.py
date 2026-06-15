import os
import sys
import cv2

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

from vision.detector_node import process_frame
from feature_matching.dino_verifier import verify

# =====================================
# Verification Threshold
# =====================================

SIMILARITY_THRESHOLD = 0.70

# =====================================
# Main Verification Function
# =====================================

def verify_targets(frame):

    detections = process_frame(
        frame
    )

    verified_targets = []

    for detection in detections:

        x1, y1, x2, y2 = detection["bbox"]

        crop = frame[
            y1:y2,
            x1:x2
        ]

        if crop.size == 0:
            continue

        dino_result = verify(
            crop
        )

        verified = (
            dino_result["similarity"]
            >=
            SIMILARITY_THRESHOLD
        )

        verified_targets.append({

            "class":
                detection["class"],

            "yolo_confidence":
                detection["confidence"],

            "dino_match":
                dino_result["match"],

            "similarity":
                dino_result["similarity"],

            "verified":
                verified,

            "center_x":
                detection["center_x"],

            "center_y":
                detection["center_y"],

            "bbox":
                detection["bbox"]
        })

    return verified_targets