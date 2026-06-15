import os
import sys
import pickle
import torch
import numpy as np

from PIL import Image
from torchvision import transforms

# =====================================
# Paths
# =====================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DATABASE_PATH = os.path.join(
    BASE_DIR,
    "embeddings",
    "dino_database.pkl"
)

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

    transforms.Resize((224, 224)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

# =====================================
# Load Database
# =====================================

with open(
    DATABASE_PATH,
    "rb"
) as f:

    database = pickle.load(f)

print(
    f"DINO Database Loaded ({len(database)} references)"
)

# =====================================
# Embedding Function
# =====================================

def get_embedding(image):

    if not isinstance(image, Image.Image):

        image = Image.fromarray(image)

    image = image.convert("RGB")

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

    return np.dot(a, b) / (
        np.linalg.norm(a)
        *
        np.linalg.norm(b)
    )

# =====================================
# Verification Function
# =====================================

def verify(crop):

    query_embedding = get_embedding(
        crop
    )

    best_class = None

    best_similarity = -1

    for name, ref_embedding in database.items():

        similarity = cosine_similarity(
            query_embedding,
            ref_embedding
        )

        if similarity > best_similarity:

            best_similarity = similarity

            best_class = name

    return {

        "match":
            best_class,

        "similarity":
            float(best_similarity)
    }