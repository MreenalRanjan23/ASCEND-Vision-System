import os
import pickle
import torch

from PIL import Image
from torchvision import transforms

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

        mean=[0.485, 0.456, 0.406],

        std=[0.229, 0.224, 0.225]
    )
])

# =====================================
# Embedding Function
# =====================================

def get_embedding(image_path):

    image = Image.open(
        image_path
    ).convert("RGB")

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
# Paths
# =====================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

IMAGE_FOLDER = os.path.join(
    BASE_DIR,
    "mission_assets",
    "reference_features"
)

DATABASE_PATH = os.path.join(
    BASE_DIR,
    "embeddings",
    "dino_database.pkl"
)

print(
    "\nIMAGE_FOLDER:",
    IMAGE_FOLDER
)

print(
    "DATABASE_PATH:",
    DATABASE_PATH
)

# =====================================
# Build Database
# =====================================

database = {}

print(
    "\nBuilding DINO Database...\n"
)

for file in os.listdir(IMAGE_FOLDER):

    image_path = os.path.join(
        IMAGE_FOLDER,
        file
    )

    try:

        embedding = get_embedding(
            image_path
        )

        database[file] = embedding

        print(
            f"Processed: {file}"
        )

    except Exception as e:

        print(
            f"Skipping {file}: {e}"
        )

# =====================================
# Save Database
# =====================================

with open(
    DATABASE_PATH,
    "wb"
) as f:

    pickle.dump(
        database,
        f
    )

print(
    "\nDINO Database Saved Successfully"
)

print(
    f"Total Images: {len(database)}"
)