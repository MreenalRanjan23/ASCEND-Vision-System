import cv2
import pickle
import torch
import numpy as np
import csv

from datetime import datetime
from PIL import Image
from torchvision import models
from torchvision import transforms

# =====================================
# Load Embedding Database
# =====================================

with open(
    "../embeddings/database.pkl",
    "rb"
) as f:

    database = pickle.load(f)

print(
    f"Database Loaded: {len(database)} images"
)

# =====================================
# MobileNetV3
# =====================================

model = models.mobilenet_v3_small(
    weights=models.MobileNet_V3_Small_Weights.DEFAULT
)

feature_extractor = torch.nn.Sequential(
    *list(model.children())[:-1]
)

feature_extractor.eval()

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
# Frame Embedding
# =====================================

def get_embedding(frame):

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    img = Image.fromarray(rgb)

    tensor = transform(img)

    tensor = tensor.unsqueeze(0)

    with torch.no_grad():

        embedding = feature_extractor(
            tensor
        )

    embedding = embedding.view(
        embedding.size(0),
        -1
    )

    return embedding.squeeze().numpy()

# =====================================
# Cosine Similarity
# =====================================

def cosine_similarity(a, b):

    return np.dot(a, b) / (
        np.linalg.norm(a)
        *
        np.linalg.norm(b)
    )

# =====================================
# Video
# =====================================

VIDEO_PATH = "../videos/test_video.mp4"

cap = cv2.VideoCapture(
    VIDEO_PATH
)

frame_count = 0

best_score = -1

best_frame_number = -1

best_frame = None

best_match_name = None

# =====================================
# Process Video
# =====================================

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_count += 1

    # Process every 30th frame

    if frame_count % 30 != 0:
        continue

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
        f"Frame {frame_count}"
    )

    print(
        f"Best Match: {best_match}"
    )

    print(
        f"Similarity: {highest_similarity:.4f}"
    )

    print("-" * 40)

    if highest_similarity > best_score:

        best_score = highest_similarity

        best_frame_number = frame_count

        best_frame = frame.copy()

        best_match_name = best_match

cap.release()

# =====================================
# Final Result
# =====================================

print("\nBEST FRAME FOUND")

print(
    f"Frame Number : {best_frame_number}"
)

print(
    f"Best Similarity : {best_score:.4f}"
)

print(
    f"Best Match : {best_match_name}"
)

# =====================================
# Save Best Frame
# =====================================

timestamp = datetime.now().strftime(
    "%Y%m%d_%H%M%S"
)

image_path = (
    f"../detections/frames/"
    f"best_frame_{timestamp}.jpg"
)

cv2.imwrite(
    image_path,
    best_frame
)

print(
    f"\nFrame Saved:"
)

print(
    image_path
)

# =====================================
# Detection Log
# =====================================

csv_path = "../detections/detection_log.csv"

with open(
    csv_path,
    "a",
    newline=""
) as file:

    writer = csv.writer(
        file
    )

    writer.writerow([
        timestamp,
        best_frame_number,
        round(best_score, 4),
        best_match_name,
        image_path
    ])

print(
    "\nDetection Logged Successfully"
)