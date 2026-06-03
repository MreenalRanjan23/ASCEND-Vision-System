import os
import pickle
import torch

from PIL import Image
from torchvision import models
from torchvision import transforms

# ==========================
# MobileNetV3
# ==========================

model = models.mobilenet_v3_small(
    weights=models.MobileNet_V3_Small_Weights.DEFAULT
)

feature_extractor = torch.nn.Sequential(
    *list(model.children())[:-1]
)

feature_extractor.eval()

# ==========================
# Transform
# ==========================

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

# ==========================
# Embedding Function
# ==========================

def get_embedding(image_path):

    img = Image.open(
        image_path
    ).convert("RGB")

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

# ==========================
# Dataset
# ==========================

DATASET_FOLDER = "../dataset/test"

database = {}

for file in os.listdir(DATASET_FOLDER):

    path = os.path.join(
        DATASET_FOLDER,
        file
    )

    try:

        embedding = get_embedding(path)

        database[file] = embedding

        print(
            f"Processed: {file}"
        )

    except Exception as e:

        print(
            f"Failed: {file}"
        )

# ==========================
# Save Database
# ==========================

with open(
    "../embeddings/database.pkl",
    "wb"
) as f:

    pickle.dump(
        database,
        f
    )

print(
    "\nDatabase Saved Successfully"
)