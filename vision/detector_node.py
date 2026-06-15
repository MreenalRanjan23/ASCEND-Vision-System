import sys
import os

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

from yolo.detector import detect


def process_frame(frame):

    detections = detect(frame)

    results = []

    for d in detections:

        results.append({

            "class":
                d["class"],

            "confidence":
                d["confidence"],

            "center_x":
                d["center"][0],

            "center_y":
                d["center"][1],

            "bbox":
                d["bbox"]
        })

    return results