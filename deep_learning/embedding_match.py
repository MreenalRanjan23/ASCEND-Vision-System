import torch
from torchvision import models
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F

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
# Function
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
# Image Paths
# =====================================

image1 = "../dataset/raw/PSP_010734_2000.jpg"

image2 = "../dataset/raw/PSP_010735_1615.jpg"

# =====================================
# Generate Embeddings
# =====================================

emb1 = get_embedding(image1)

emb2 = get_embedding(image2)

# =====================================
# Cosine Similarity
# =====================================

similarity = F.cosine_similarity(
    emb1,
    emb2
)

print(
    "Similarity Score:",
    similarity.item()
)