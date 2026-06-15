import torch
import torch.nn.functional as F

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

    image = Image.open(
        image_path
    ).convert("RGB")

    tensor = transform(
        image
    )

    tensor = tensor.unsqueeze(
        0
    )

    with torch.no_grad():

        embedding = model(
            tensor
        )

    return embedding

# =====================================
# Images
# =====================================

IMAGE_1 = "../dataset/raw/PSP_010734_2000.jpg"

IMAGE_2 = "../dataset/raw/PSP_010880_2255.jpg"

# Try changing IMAGE_2 later

# =====================================
# Embeddings
# =====================================

embedding_1 = get_embedding(
    IMAGE_1
)

embedding_2 = get_embedding(
    IMAGE_2
)

# =====================================
# Similarity
# =====================================

similarity = F.cosine_similarity(

    embedding_1,
    embedding_2

).item()

print(
    "\nSimilarity Score:"
)

print(
    similarity
)