import os
import cv2
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
# ORB Verification Function
# =====================================

def orb_score(image1_path, image2_path):

    img1 = cv2.imread(image1_path)
    img2 = cv2.imread(image2_path)

    orb = cv2.ORB_create(
        nfeatures=1000
    )

    kp1, des1 = orb.detectAndCompute(
        img1,
        None
    )

    kp2, des2 = orb.detectAndCompute(
        img2,
        None
    )

    if des1 is None or des2 is None:
        return 0

    bf = cv2.BFMatcher(
        cv2.NORM_HAMMING
    )

    matches = bf.knnMatch(
        des1,
        des2,
        k=2
    )

    good_matches = []

    for m, n in matches:

        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    return len(good_matches)


# =====================================
# Paths
# =====================================

REFERENCE_IMAGE = "../dataset/reference/target.jpg"

TEST_FOLDER = "../dataset/test"

# =====================================
# Generate Reference Embedding
# =====================================

reference_embedding = get_embedding(
    REFERENCE_IMAGE
)

# =====================================
# Compare Against Dataset
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

        orb_matches = orb_score(
            REFERENCE_IMAGE,
            image_path
        )

        results.append(
            (
                file,
                similarity,
                orb_matches
            )
        )

    except Exception as e:

        print(
            f"Skipping {file}: {e}"
        )

# =====================================
# Sort By Similarity
# =====================================

results.sort(
    key=lambda x: x[1],
    reverse=True
)

# =====================================
# Display Results
# =====================================

print("\nTop Matches:\n")

for rank, (
    filename,
    similarity,
    orb_matches
) in enumerate(
    results[:10],
    start=1
):

    print(
        f"{rank}. {filename}"
    )

    print(
        f"   Similarity : {similarity:.4f}"
    )

    print(
        f"   ORB Matches: {orb_matches}"
    )

    print("-" * 40)