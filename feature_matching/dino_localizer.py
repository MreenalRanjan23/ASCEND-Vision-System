import cv2
import pickle
import torch
import numpy as np

from PIL import Image
from torchvision import transforms

# =====================================
# Reference Image
# =====================================

REFERENCE_IMAGE = "../dataset/reference/target.jpg"

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

def get_embedding(image):

    if isinstance(image, str):

        image = Image.open(
            image
        ).convert("RGB")

    else:

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        image = Image.fromarray(
            image
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
# Cosine Similarity
# =====================================

def cosine_similarity(a, b):

    return np.dot(a,b) / (
        np.linalg.norm(a)
        *
        np.linalg.norm(b)
    )

# =====================================
# Reference Embedding
# =====================================

print(
    "Generating Reference Embedding..."
)

reference_embedding = get_embedding(
    REFERENCE_IMAGE
)

print(
    "Reference Loaded\n"
)

# =====================================
# Camera
# =====================================

cap = cv2.VideoCapture(1)

if not cap.isOpened():

    print(
        "Could not open camera"
    )

    exit()

# =====================================
# Grid Settings
# =====================================

GRID_SIZE = 5

# =====================================
# Main Loop
# =====================================

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_height = frame.shape[0]
    frame_width = frame.shape[1]

    patch_h = frame_height // GRID_SIZE
    patch_w = frame_width // GRID_SIZE

    best_similarity = -1

    best_box = None

    # ==========================
    # Scan Patches
    # ==========================

    for row in range(GRID_SIZE):

        for col in range(GRID_SIZE):

            x1 = col * patch_w
            y1 = row * patch_h

            x2 = x1 + patch_w
            y2 = y1 + patch_h

            patch = frame[
                y1:y2,
                x1:x2
            ]

            try:

                patch_embedding = get_embedding(
                    patch
                )

                similarity = cosine_similarity(
                    reference_embedding,
                    patch_embedding
                )

                if similarity > best_similarity:

                    best_similarity = similarity

                    best_box = (
                        x1,
                        y1,
                        x2,
                        y2
                    )

            except:
                pass

    # ==========================
    # Draw Box
    # ==========================

    if best_box is not None:

        x1,y1,x2,y2 = best_box

        cv2.rectangle(
            frame,
            (x1,y1),
            (x2,y2),
            (0,255,0),
            3
        )

        center_x = (
            x1 + x2
        ) // 2

        center_y = (
            y1 + y2
        ) // 2

        cv2.circle(
            frame,
            (center_x,center_y),
            5,
            (0,0,255),
            -1
        )

        cv2.putText(
            frame,
            f"Similarity: {best_similarity:.3f}",
            (20,40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0,255,0),
            2
        )

        cv2.putText(
            frame,
            f"Center: ({center_x},{center_y})",
            (20,80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0,255,0),
            2
        )

    cv2.imshow(
        "DINO Localizer",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# =====================================
# Cleanup
# =====================================

cap.release()

cv2.destroyAllWindows()