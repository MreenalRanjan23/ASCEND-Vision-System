import os
import cv2

# =====================================
# Paths
# =====================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DETECTIONS_DIR = os.path.join(
    BASE_DIR,
    "detections"
)

os.makedirs(
    DETECTIONS_DIR,
    exist_ok=True
)

# =====================================
# Found Targets Database
# =====================================

found_targets = {}

# =====================================
# Mission Manager
# =====================================

def process_verified_targets(
    frame,
    targets
):

    global found_targets

    for target in targets:

        # -----------------------------
        # Skip Unverified Targets
        # -----------------------------

        if not target["verified"]:
            continue

        class_name = target["class"]

        # -----------------------------
        # Skip Duplicates
        # -----------------------------

        if class_name in found_targets:

            continue

        # -----------------------------
        # Extract Bounding Box
        # -----------------------------

        x1, y1, x2, y2 = target["bbox"]

        crop = frame[
            y1:y2,
            x1:x2
        ]

        # -----------------------------
        # Save Detection Image
        # -----------------------------

        image_path = os.path.join(
            DETECTIONS_DIR,
            f"{class_name}.jpg"
        )

        cv2.imwrite(
            image_path,
            crop
        )

        # -----------------------------
        # Save Metadata
        # -----------------------------

        found_targets[class_name] = {

            "class":
                class_name,

            "yolo_confidence":
                target["yolo_confidence"],

            "dino_match":
                target["dino_match"],

            "similarity":
                target["similarity"],

            "center_x":
                target["center_x"],

            "center_y":
                target["center_y"],

            "image":
                image_path
        }

        # -----------------------------
        # Console Output
        # -----------------------------

        print("\n====================")

        print(
            f"NEW TARGET FOUND: {class_name}"
        )

        print(
            f"YOLO Confidence: "
            f"{target['yolo_confidence']:.2f}"
        )

        print(
            f"DINO Similarity: "
            f"{target['similarity']:.2f}"
        )

        print(
            f"Saved Image: {image_path}"
        )

        print(
            "====================\n"
        )

    return found_targets