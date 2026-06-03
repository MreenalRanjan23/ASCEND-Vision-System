import os
import torch
import torch.nn.functional as F

from torchvision import models
from torchvision import transforms

from PIL import Image

# =====================================
# Load MobileNetV3
# =====================================

model = models.mobilenet_v3_small(
    weights=models.MobileNet_V3_Small_Weights.DEFAULT
)

feature_extractor = torch.nn.Sequential(
    *list(model.children())[:-1]
)

feature_extractor.eval()

# =====================================
# Image Transform
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

def get_embedding(image_path):

    img = Image.open(
        image_path
    ).convert("RGB")

    img_tensor = transform(img)

    img_tensor = img_tensor.unsqueeze(0)

    with torch.no_grad():

        embedding = feature_extractor(
            img_tensor
        )

    embedding = embedding.view(
        embedding.size(0),
        -1
    )

    return embedding


# =====================================
# Paths
# =====================================

REFERENCE_IMAGE = "../dataset/reference/target.jpg"

TEST_FOLDER = "../dataset/test"

# =====================================
# Reference Embedding
# =====================================

reference_embedding = get_embedding(
    REFERENCE_IMAGE
)

# =====================================
# Compare Against All Images
# =====================================

results = []

for file in os.listdir(TEST_FOLDER):

    image_path = os.path.join(
        TEST_FOLDER,
        file
    )

    try:

        embedding = get_embedding(
            image_path
        )

        similarity = F.cosine_similarity(
            reference_embedding,
            embedding
        ).item()

        results.append(
            (
                file,
                similarity
            )
        )

    except Exception as e:

        print(
            f"Skipping {file}: {e}"
        )

# =====================================
# Sort Results
# =====================================

results.sort(
    key=lambda x: x[1],
    reverse=True
)

# =====================================
# Display Top Matches
# =====================================

print("\nTop Matches:\n")

for rank, (filename, score) in enumerate(
    results[:10],
    start=1
):

    print(
        f"{rank}. {filename} -> {score:.4f}"
    )