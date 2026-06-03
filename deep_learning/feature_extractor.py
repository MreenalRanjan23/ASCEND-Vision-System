import torch
from torchvision import models
from PIL import Image
from torchvision import transforms

# =====================================
# Load Image
# =====================================

img = Image.open(
    "../dataset/raw/PSP_010734_2000.jpg"
).convert("RGB")

# =====================================
# Load Pretrained MobileNetV3
# =====================================

model = models.mobilenet_v3_small(
    weights=models.MobileNet_V3_Small_Weights.DEFAULT
)

# =====================================
# Remove Final Classifier Layer
# =====================================

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
# Convert Image to Tensor
# =====================================

img_tensor = transform(img)

print("Tensor Shape Before Batch:", img_tensor.shape)

img_tensor = img_tensor.unsqueeze(0)

print("Tensor Shape After Batch:", img_tensor.shape)

# =====================================
# Generate Embedding
# =====================================

with torch.no_grad():

    embedding = feature_extractor(
        img_tensor
    )

print("\nEmbedding Shape:")
print(embedding.shape)

# =====================================
# Flatten Embedding
# =====================================

embedding_vector = embedding.view(
    embedding.size(0),
    -1
)

print("\nFlattened Embedding Shape:")
print(embedding_vector.shape)

# =====================================
# Print First Few Values
# =====================================

print("\nFirst 20 Embedding Values:")

print(
    embedding_vector[0][:20]
)