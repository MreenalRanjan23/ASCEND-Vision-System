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

    transforms.Resize((224,224)),

    transforms.ToTensor(),

    transforms.Normalize(

        mean=[0.485,0.456,0.406],

        std=[0.229,0.224,0.225]
    )
])

# =====================================
# Load Image
# =====================================

image = Image.open(
    "../dataset/raw/PSP_010734_2000.jpg"
).convert("RGB")

# =====================================
# Convert To Tensor
# =====================================

image_tensor = transform(
    image
)

print(
    "Tensor Shape Before Batch:"
)

print(
    image_tensor.shape
)

# =====================================
# Add Batch Dimension
# =====================================

image_tensor = image_tensor.unsqueeze(
    0
)

print(
    "\nTensor Shape After Batch:"
)

print(
    image_tensor.shape
)

# =====================================
# Extract Embedding
# =====================================

with torch.no_grad():

    embedding = model(
        image_tensor
    )

print(
    "\nEmbedding Shape:"
)

print(
    embedding.shape
)

print(
    "\nFirst 20 Values:"
)

print(
    embedding[0][:20]
)