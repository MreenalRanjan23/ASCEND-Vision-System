import cv2
import pickle
import torch
import numpy as np

from PIL import Image
from torchvision import transforms

# =====================================
# Load DINO Database
# =====================================

with open(
    "../embeddings/dino_database.pkl",
    "rb"
) as f:

    database = pickle.load(f)

print(
    f"Database Loaded: {len(database)} images"
)

# =====================================
# Detection Threshold
# =====================================

CONFIDENCE_THRESHOLD = 0.30

LOCK_THRESHOLD = 5

# =====================================
# Load DINOv2
# =====================================

model = torch.hub.load(
    "facebookresearch/dinov2",
    "dinov2_vits14"
)

model.eval()

# =====================================
# Transform
# =====================================

transform = transforms.Compose([

    transforms.Resize((224,224)),

    transforms.ToTensor(),

    transforms.Normalize(

        mean=[0.485,0.456,0.406],

        std=[0.229,0.224,0.225]
    )
])

# =====================================
# Embedding Function
# =====================================

def get_embedding(frame):

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    image = Image.fromarray(
        rgb
    )

    tensor = transform(
        image
    )

    tensor = tensor.unsqueeze(0)

    with torch.no_grad():

        embedding = model(
            tensor
        )

    return embedding.squeeze().numpy()

# =====================================
# Similarity
# =====================================

def cosine_similarity(a, b):

    return np.dot(a, b) / (
        np.linalg.norm(a)
        *
        np.linalg.norm(b)
    )

# =====================================
# Open DroidCam
# =====================================

cap = cv2.VideoCapture(1)

if not cap.isOpened():

    print(
        "Could not open DroidCam"
    )

    exit()

print(
    "\nDINO LIVE DETECTION STARTED\n"
)

frame_count = 0

highest_similarity = 0

best_match = "None"

consecutive_detections = 0

target_locked = False

# =====================================
# Live Loop
# =====================================

while True:

    ret, frame = cap.read()

    if not ret:

        print(
            "No frame received"
        )

        break

    frame_count += 1

    # =====================================
    # Process Every 15 Frames
    # =====================================

    if frame_count % 15 == 0:

        frame_embedding = get_embedding(
            frame
        )

        highest_similarity = -1

        best_match = None

        for filename, db_embedding in database.items():

            similarity = cosine_similarity(
                frame_embedding,
                db_embedding
            )

            if similarity > highest_similarity:

                highest_similarity = similarity

                best_match = filename

        print(
            f"Best Match: {best_match}"
        )

        print(
            f"Similarity: {highest_similarity:.4f}"
        )

        print(
            "-" * 40
        )

        # =====================================
        # Detection Stabilizer
        # =====================================

        if highest_similarity >= CONFIDENCE_THRESHOLD:

            consecutive_detections += 1

        else:

            consecutive_detections = 0

            target_locked = False

        # =====================================
        # Target Lock
        # =====================================

        if consecutive_detections >= LOCK_THRESHOLD:
            consecutive_detections = LOCK_THRESHOLD

            target_locked = True

    # =====================================
    # Status Logic
    # =====================================

    if target_locked:

        status_text = (
            f"TARGET LOCKED"
        )

        status_color = (
            0,
            255,
            0
        )

    elif highest_similarity >= CONFIDENCE_THRESHOLD:

        status_text = (
            f"TARGET FOUND"
        )

        status_color = (
            0,
            255,
            255
        )

    else:

        status_text = (
            "NO TARGET"
        )

        status_color = (
            0,
            0,
            255
        )

    # =====================================
    # Overlay
    # =====================================

    cv2.putText(
        frame,
        status_text,
        (20,40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        status_color,
        2
    )

    cv2.putText(
        frame,
        f"Similarity: {highest_similarity:.4f}",
        (20,80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255,255,255),
        2
    )

    cv2.putText(
        frame,
        f"Lock Count: {consecutive_detections}/{LOCK_THRESHOLD}",
        (20,120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255,255,255),
        2
    )

    cv2.putText(
        frame,
        f"Match: {best_match}",
        (20,160),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255,255,255),
        2
    )

    cv2.imshow(
        "ASCEND DINO Live Detection",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord('q'):

        break

# =====================================
# Cleanup
# =====================================

cap.release()

cv2.destroyAllWindows()